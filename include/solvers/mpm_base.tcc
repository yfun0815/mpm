//! Constructor
template <unsigned Tdim>
mpm::MPMBase<Tdim>::MPMBase(const std::shared_ptr<IO>& io) : mpm::MPM(io) {
  //! Logger
  console_ = spdlog::get("MPMBase");

  // Create a mesh with global id 0
  const mpm::Index id = 0;

  // Set analysis step to start at 0
  step_ = 0;

  // Set mesh as isoparametric
  bool isoparametric = is_isoparametric();

  // Construct mesh, use levelset mesh if levelset active
  if (is_levelset()) {
    mesh_ = std::make_shared<mpm::MeshLevelset<Tdim>>(id, isoparametric);
  } else {
    mesh_ = std::make_shared<mpm::Mesh<Tdim>>(id, isoparametric);
  }

  // Create constraints
  constraints_ = std::make_shared<mpm::Constraints<Tdim>>(mesh_);

  // Empty all materSials
  materials_.clear();

  // Variable list
  tsl::robin_map<std::string, VariableType> variables = {
      // Scalar variables
      {"id", VariableType::Scalar},
      {"material", VariableType::Scalar},
      {"mass", VariableType::Scalar},
      {"volume", VariableType::Scalar},
      {"mass_density", VariableType::Scalar},
      {"levelset", VariableType::Scalar},
      // Vector variables
      {"displacements", VariableType::Vector},
      {"velocities", VariableType::Vector},
      {"normals", VariableType::Vector},
      {"levelset_couples", VariableType::Vector},
      // Tensor variables
      {"strains", VariableType::Tensor},
      {"stresses", VariableType::Tensor}};

  try {
    analysis_ = io_->analysis();
    // Time-step size
    dt_ = analysis_["dt"].template get<double>();
    // Number of time steps
    nsteps_ = analysis_["nsteps"].template get<mpm::Index>();

    // nload balance
    if (analysis_.find("nload_balance_steps") != analysis_.end())
      nload_balance_steps_ =
          analysis_["nload_balance_steps"].template get<mpm::Index>();

    // Locate particles
    if (analysis_.find("locate_particles") != analysis_.end())
      locate_particles_ = analysis_["locate_particles"].template get<bool>();

    // Stress update method (USF/USL/MUSL/Newmark)
    try {
      if (analysis_.find("mpm_scheme") != analysis_.end())
        stress_update_ = analysis_["mpm_scheme"].template get<std::string>();
      else
        throw std::runtime_error("\"mpm_scheme\" is undefined");
    } catch (std::exception& exception) {
      console_->warn(
          "{} #{}: Stress update method is not specified, using \"usf\" as "
          "default; {}",
          __FILE__, __LINE__, exception.what());
    }

    // Velocity update
    std::string vel_update_type;
    try {
      if (analysis_.find("velocity_update") != analysis_.end()) {
        if (analysis_["velocity_update"].is_boolean()) {
          bool v_update = analysis_["velocity_update"].template get<bool>();
          vel_update_type = (v_update) ? "pic" : "flip";
        } else
          vel_update_type =
              analysis_["velocity_update"].template get<std::string>();
      } else
        throw std::runtime_error("\"velocity_update\" is undefined");

      // Check if blending_ratio is specified
      if (analysis_.contains("velocity_update_settings") &&
          analysis_["velocity_update_settings"].contains("blending_ratio")) {
        blending_ratio_ = analysis_["velocity_update_settings"]
                              .at("blending_ratio")
                              .template get<double>();
        // Check if blending ratio value is appropriately assigned
        if (blending_ratio_ < 0. || blending_ratio_ > 1.) {
          blending_ratio_ = 1.0;
          console_->warn(
              "{} #{}: FLIP-PIC Blending ratio is not properly assigned, using "
              "1.0 as default",
              __FILE__, __LINE__);
        }
      }

    } catch (std::exception& exception) {
      console_->warn(
          "{} #{}: Velocity update method is not properly specified, using "
          "\"flip\" as default; {}",
          __FILE__, __LINE__, exception.what());
      vel_update_type = "flip";
    }
    velocity_update_ = VelocityUpdateType.at(vel_update_type);

    // Damping
    try {
      if (analysis_.find("damping") != analysis_.end()) {
        if (!initialise_damping(analysis_.at("damping")))
          throw std::runtime_error("Damping parameters are undefined");
      } else
        throw std::runtime_error("\"damping\" is undefined");
    } catch (std::exception& exception) {
      console_->warn(
          "{} #{}: Damping is not specified, using \"None\" as default; {}",
          __FILE__, __LINE__, exception.what());
    }

    // Math functions
    try {
      // Get materials properties
      auto math_functions = io_->json_object("math_functions");
      if (!math_functions.empty())
        this->initialise_math_functions(math_functions);
      else
        throw std::runtime_error("");
    } catch (std::exception& exception) {
      console_->warn(
          "{} #{}: Math functions are undefined; Math functions JSON data not "
          "found",
          __FILE__, __LINE__);
    }

    post_process_ = io_->post_processing();
    // Output steps
    output_steps_ = post_process_["output_steps"].template get<mpm::Index>();

  } catch (std::domain_error& domain_error) {
    console_->error("{} #{}: Get analysis object \"{}\"", __FILE__, __LINE__,
                    domain_error.what());
    abort();
  }

  // VTK particle variables
  // Initialise container with empty vector
  vtk_vars_.insert(
      std::make_pair(mpm::VariableType::Scalar, std::vector<std::string>()));
  vtk_vars_.insert(
      std::make_pair(mpm::VariableType::Vector, std::vector<std::string>()));
  vtk_vars_.insert(
      std::make_pair(mpm::VariableType::Tensor, std::vector<std::string>()));

  if ((post_process_.find("vtk") != post_process_.end()) &&
      post_process_.at("vtk").is_array() &&
      post_process_.at("vtk").size() > 0) {
    // Iterate over vtk
    for (unsigned i = 0; i < post_process_.at("vtk").size(); ++i) {
      std::string attribute =
          post_process_["vtk"][i].template get<std::string>();
      if (attribute == "geometry")
        geometry_vtk_ = true;
      else if (variables.find(attribute) != variables.end())
        vtk_vars_[variables.at(attribute)].emplace_back(attribute);
      else {
        console_->warn(
            "{} #{}: VTK variable \"{}\" was specified, but is not available "
            "in variable list",
            __FILE__, __LINE__, attribute);
      }
    }
  } else {
    console_->warn(
        "{} #{}: No VTK variables were specified, none will be generated",
        __FILE__, __LINE__);
  }

  // VTK state variables
  bool vtk_statevar = false;
  if ((post_process_.find("vtk_statevars") != post_process_.end()) &&
      post_process_.at("vtk_statevars").is_array() &&
      post_process_.at("vtk_statevars").size() > 0) {
    // Iterate over state_vars
    for (const auto& svars : post_process_["vtk_statevars"]) {
      // Phase id
      unsigned phase_id = 0;
      if (svars.contains("phase_id"))
        phase_id = svars.at("phase_id").template get<unsigned>();

      // State variables
      if (svars.at("statevars").is_array() &&
          svars.at("statevars").size() > 0) {
        // Insert vtk_statevars_
        const std::vector<std::string> state_var = svars["statevars"];
        vtk_statevars_.insert(std::make_pair(phase_id, state_var));
        vtk_statevar = true;
      } else {
        vtk_statevar = false;
        break;
      }
    }
  }
  if (!vtk_statevar)
    console_->warn(
        "{} #{}: No VTK statevariable were specified, none will be generated",
        __FILE__, __LINE__);
}

// Initialise mesh
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_mesh() {
  // Initialise MPI rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Get mesh properties
  auto mesh_props = io_->json_object("mesh");
  // Get Mesh reader from JSON object
  const std::string io_type = mesh_props["io_type"].template get<std::string>();

  bool check_duplicates = true;
  try {
    check_duplicates = mesh_props["check_duplicates"].template get<bool>();
  } catch (std::exception& exception) {
    console_->warn(
        "{} #{}: Check duplicates not specified, using \"true\" as default; {}",
        __FILE__, __LINE__, exception.what());
    check_duplicates = true;
  }

  // Create a mesh reader
  auto mesh_io = Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

  auto nodes_begin = std::chrono::steady_clock::now();
  // Global Index
  mpm::Index gid = 0;
  // Node type
  const auto node_type = mesh_props["node_type"].template get<std::string>();

  // Mesh file
  std::string mesh_file =
      io_->file_name(mesh_props["mesh"].template get<std::string>());

  // Create nodes from file
  bool node_status =
      mesh_->create_nodes(gid,                                  // global id
                          node_type,                            // node type
                          mesh_io->read_mesh_nodes(mesh_file),  // coordinates
                          check_duplicates);                    // check dups

  if (!node_status)
    throw std::runtime_error(
        "mpm::base::initialise_mesh(): Addition of nodes to mesh failed");

  auto nodes_end = std::chrono::steady_clock::now();
  console_->info("Rank {} Read nodes: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     nodes_end - nodes_begin)
                     .count());

  // Read and assign node sets
  this->node_entity_sets(mesh_props, check_duplicates);

  // Read nodal euler angles and assign rotation matrices
  this->node_euler_angles(mesh_props, mesh_io);

  // Read and assign velocity constraints
  this->nodal_velocity_constraints(mesh_props, mesh_io);

  // Read and assign acceleration constraints
  this->nodal_acceleration_constraints(mesh_props, mesh_io);

  // Read and assign velocity constraints for implicit solver
  this->nodal_displacement_constraints(mesh_props, mesh_io);

  // Read and assign friction constraints
  this->nodal_frictional_constraints(mesh_props, mesh_io);

  // Read and assign adhesion constraints
  this->nodal_adhesional_constraints(mesh_props, mesh_io);

  // Read and assign pressure constraints
  this->nodal_pressure_constraints(mesh_props, mesh_io);

  // Read and assign absorbing constraintes
  this->nodal_absorbing_constraints(mesh_props, mesh_io);

  // Read and assign interface (includes multimaterial and levelset)
  this->interface_inputs(mesh_props, mesh_io);

  // Initialise cell
  auto cells_begin = std::chrono::steady_clock::now();
  // Shape function name
  const auto cell_type = mesh_props["cell_type"].template get<std::string>();
  // Shape function
  std::shared_ptr<mpm::Element<Tdim>> element =
      Factory<mpm::Element<Tdim>>::instance()->create(cell_type);

  // Create cells from file
  bool cell_status =
      mesh_->create_cells(gid,                                  // global id
                          element,                              // element type
                          mesh_io->read_mesh_cells(mesh_file),  // Node ids
                          check_duplicates);                    // Check dups

  if (!cell_status)
    throw std::runtime_error(
        "mpm::base::initialise_mesh(): Addition of cells to mesh failed");

  // Compute cell neighbours
  mesh_->find_cell_neighbours();

  // Read and assign cell sets
  this->cell_entity_sets(mesh_props, check_duplicates);

  // Use Nonlocal basis
  if (cell_type.back() == 'B' || cell_type.back() == 'L') {
    this->initialise_nonlocal_mesh(mesh_props);
  }

  auto cells_end = std::chrono::steady_clock::now();
  console_->info("Rank {} Read cells: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     cells_end - cells_begin)
                     .count());
}

// Initialise particle types
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_particle_types() {
  // Get particles properties
  auto json_particles = io_->json_object("particles");

  for (const auto& json_particle : json_particles) {
    // Gather particle types
    auto particle_type =
        json_particle["generator"]["particle_type"].template get<std::string>();
    particle_types_.insert(particle_type);
  }
}

// Initialise particles
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_particles() {
  // Initialise MPI rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Get mesh properties
  auto mesh_props = io_->json_object("mesh");
  // Get Mesh reader from JSON object
  const std::string io_type = mesh_props["io_type"].template get<std::string>();

  // Check duplicates default set to true
  bool check_duplicates = true;
  if (mesh_props.find("check_duplicates") != mesh_props.end())
    check_duplicates = mesh_props["check_duplicates"].template get<bool>();

  auto particles_gen_begin = std::chrono::steady_clock::now();

  // Get particles properties
  auto json_particles = io_->json_object("particles");

  for (const auto& json_particle : json_particles) {
    // Generate particles
    bool gen_status =
        mesh_->generate_particles(io_, json_particle["generator"]);
    if (!gen_status)
      std::runtime_error(
          "mpm::base::initialise_particles() Generate particles failed");
  }

  // Gather particle types
  this->initialise_particle_types();

  auto particles_gen_end = std::chrono::steady_clock::now();
  console_->info("Rank {} Generate particles: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     particles_gen_end - particles_gen_begin)
                     .count());

  auto particles_locate_begin = std::chrono::steady_clock::now();

  // Create a mesh reader
  auto particle_io = Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

  // Read and assign particles cells
  this->particles_cells(mesh_props, particle_io);

  // Locate particles in cell
  auto unlocatable_particles = mesh_->locate_particles_mesh();

  if (!unlocatable_particles.empty())
    throw std::runtime_error(
        "mpm::base::initialise_particles() Particle outside the mesh domain");

  // Write particles and cells to file
  particle_io->write_particles_cells(
      io_->output_file("particles-cells", ".txt", uuid_, 0, 0).string(),
      mesh_->particles_cells());

  auto particles_locate_end = std::chrono::steady_clock::now();
  console_->info("Rank {} Locate particles: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     particles_locate_end - particles_locate_begin)
                     .count());

  auto particles_volume_begin = std::chrono::steady_clock::now();
  // Compute volume
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_volume, std::placeholders::_1));

  // Read and assign particles volumes
  this->particles_volumes(mesh_props, particle_io);

  // Read and assign particles stresses
  this->particles_stresses(mesh_props, particle_io);

  // Read and assign particles initial pore pressure
  this->particles_pore_pressures(mesh_props, particle_io);

  auto particles_volume_end = std::chrono::steady_clock::now();
  console_->info("Rank {} Read volume, velocity and stresses: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     particles_volume_end - particles_volume_begin)
                     .count());

  // Particle entity sets
  this->particle_entity_sets(check_duplicates);

  // Read and assign particles velocity constraints
  this->particle_velocity_constraints();

  console_->info("Rank {} Create particle sets: {} ms", mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     particles_volume_end - particles_volume_begin)
                     .count());

  // Material id update using particle sets
  try {
    auto material_sets = io_->json_object("material_sets");
    if (!material_sets.empty()) {
      for (const auto& material_set : material_sets) {
        unsigned material_id =
            material_set["material_id"].template get<unsigned>();
        unsigned phase_id = mpm::ParticlePhase::Solid;
        if (material_set.contains("phase_id"))
          phase_id = material_set["phase_id"].template get<unsigned>();
        unsigned pset_id = material_set["pset_id"].template get<unsigned>();
        // Update material_id for particles in each pset
        mesh_->iterate_over_particle_set(
            pset_id, std::bind(&mpm::ParticleBase<Tdim>::assign_material,
                               std::placeholders::_1,
                               materials_.at(material_id), phase_id));
      }
    } else
      throw std::runtime_error("");
  } catch (std::exception& exception) {
    console_->warn(
        "{} #{}: Material sets are undefined; Material sets JSON data not "
        "found",
        __FILE__, __LINE__);
  }
}

// Initialise materials
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_materials() {
  // Get materials properties
  auto materials = io_->json_object("materials");

  for (const auto material_props : materials) {
    // Get material type
    const std::string material_type =
        material_props["type"].template get<std::string>();

    // Get material id
    auto material_id = material_props["id"].template get<unsigned>();

    // Create a new material from JSON object
    auto mat =
        Factory<mpm::Material<Tdim>, unsigned, const Json&>::instance()->create(
            material_type, std::move(material_id), material_props);

    // Add material to list
    auto result = materials_.insert(std::make_pair(mat->id(), mat));

    // If insert material failed
    if (!result.second)
      throw std::runtime_error(
          "mpm::base::initialise_materials(): New material cannot be added, "
          "insertion failed");
  }
  // Copy materials to mesh
  mesh_->initialise_material_models(this->materials_);
}

//! Checkpoint resume
template <unsigned Tdim>
bool mpm::MPMBase<Tdim>::checkpoint_resume() {
  bool checkpoint = true;
  try {
    int mpi_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

    // Gather particle types
    this->initialise_particle_types();

    if (!analysis_["resume"]["resume"].template get<bool>())
      throw std::runtime_error("Resume analysis option is disabled");

    // Get unique analysis id
    this->uuid_ = analysis_["resume"]["uuid"].template get<std::string>();
    // Get step
    this->step_ = analysis_["resume"]["step"].template get<mpm::Index>();

    // Input particle h5 file for resume
    for (const auto ptype : particle_types_) {
      std::string attribute = mpm::ParticlePODTypeName.at(ptype);
      std::string extension = ".h5";

      auto particles_file =
          io_->output_file(attribute, extension, uuid_, step_, this->nsteps_)
              .string();

      // Load particle information from file
      mesh_->read_particles_hdf5(particles_file, attribute, ptype);
    }

    // Clear all particle ids
    mesh_->iterate_over_cells(
        std::bind(&mpm::Cell<Tdim>::clear_particle_ids, std::placeholders::_1));

    // Locate particles
    auto unlocatable_particles = mesh_->locate_particles_mesh();

    if (!unlocatable_particles.empty())
      throw std::runtime_error("Particle outside the mesh domain");

    console_->info("Checkpoint resume at step {} of {}", this->step_,
                   this->nsteps_);

  } catch (std::exception& exception) {
    console_->info("{} #{}: Resume failed, restarting analysis; {}", __FILE__,
                   __LINE__, exception.what());
    this->step_ = 0;
    checkpoint = false;
  }
  return checkpoint;
}

//! Write HDF5 files
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::write_hdf5(mpm::Index step, mpm::Index max_steps) {
  // Write hdf5 file for single phase particle
  for (const auto ptype : particle_types_) {
    std::string attribute = mpm::ParticlePODTypeName.at(ptype);
    std::string extension = ".h5";

    auto particles_file =
        io_->output_file(attribute, extension, uuid_, step, max_steps).string();

    // Load particle information from file
    if (attribute == "particles" || attribute == "fluid_particles")
      mesh_->write_particles_hdf5(particles_file);
    else if (attribute == "twophase_particles")
      mesh_->write_particles_hdf5_twophase(particles_file);
  }
}

#ifdef USE_VTK
//! Write VTK files
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::write_vtk(mpm::Index step, mpm::Index max_steps) {

  // VTK PolyData writer
  auto vtk_writer = std::make_unique<VtkWriter>(mesh_->particle_coordinates());
  const std::string extension = ".vtp";

  // Write mesh on step 0
  // Get active node pairs use true
  if (step % nload_balance_steps_ == 0)
    vtk_writer->write_mesh(
        io_->output_file("mesh", extension, uuid_, step, max_steps).string(),
        mesh_->nodal_coordinates(), mesh_->node_pairs(true));

  // VTK geometry
  if (geometry_vtk_) {
    auto meshfile =
        io_->output_file("geometry", extension, uuid_, step, max_steps)
            .string();
    vtk_writer->write_geometry(meshfile);
  }

  // MPI parallel vtk file
  int mpi_rank = 0;
  int mpi_size = 1;
  bool write_mpi_rank = false;

#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  //! VTK scalar variables
  for (const auto& attribute : vtk_vars_.at(mpm::VariableType::Scalar)) {
    // Write scalar
    auto file =
        io_->output_file(attribute, extension, uuid_, step, max_steps).string();
    vtk_writer->write_scalar_point_data(
        file, mesh_->particles_scalar_data(attribute), attribute);

    // Write a parallel MPI VTK container file
#ifdef USE_MPI
    if (mpi_rank == 0 && mpi_size > 1) {
      auto parallel_file = io_->output_file(attribute, ".pvtp", uuid_, step,
                                            max_steps, write_mpi_rank)
                               .string();

      vtk_writer->write_parallel_vtk(parallel_file, attribute, mpi_size, step,
                                     max_steps, 1);
    }
#endif
  }

  //! VTK vector variables
  for (const auto& attribute : vtk_vars_.at(mpm::VariableType::Vector)) {
    // Write vector
    auto file =
        io_->output_file(attribute, extension, uuid_, step, max_steps).string();
    vtk_writer->write_vector_point_data(
        file, mesh_->particles_vector_data(attribute), attribute);

    // Write a parallel MPI VTK container file
#ifdef USE_MPI
    if (mpi_rank == 0 && mpi_size > 1) {
      auto parallel_file = io_->output_file(attribute, ".pvtp", uuid_, step,
                                            max_steps, write_mpi_rank)
                               .string();

      vtk_writer->write_parallel_vtk(parallel_file, attribute, mpi_size, step,
                                     max_steps, 3);
    }
#endif
  }

  //! VTK tensor variables
  for (const auto& attribute : vtk_vars_.at(mpm::VariableType::Tensor)) {
    // Write vector
    auto file =
        io_->output_file(attribute, extension, uuid_, step, max_steps).string();
    vtk_writer->write_tensor_point_data(
        file, mesh_->template particles_tensor_data<6>(attribute), attribute);

    // Write a parallel MPI VTK container file
#ifdef USE_MPI
    if (mpi_rank == 0 && mpi_size > 1) {
      auto parallel_file = io_->output_file(attribute, ".pvtp", uuid_, step,
                                            max_steps, write_mpi_rank)
                               .string();

      vtk_writer->write_parallel_vtk(parallel_file, attribute, mpi_size, step,
                                     max_steps, 9);
    }
#endif
  }

  // VTK state variables
  for (auto const& vtk_statevar : vtk_statevars_) {
    unsigned phase_id = vtk_statevar.first;
    for (const auto& attribute : vtk_statevar.second) {
      std::string phase_attribute =
          "phase" + std::to_string(phase_id) + attribute;
      // Write state variables
      auto file =
          io_->output_file(phase_attribute, extension, uuid_, step, max_steps)
              .string();
      vtk_writer->write_scalar_point_data(
          file, mesh_->particles_statevars_data(attribute, phase_id),
          phase_attribute);
      // Write a parallel MPI VTK container file
#ifdef USE_MPI
      if (mpi_rank == 0 && mpi_size > 1) {
        auto parallel_file = io_->output_file(phase_attribute, ".pvtp", uuid_,
                                              step, max_steps, write_mpi_rank)
                                 .string();
        unsigned ncomponents = 1;
        vtk_writer->write_parallel_vtk(parallel_file, phase_attribute, mpi_size,
                                       step, max_steps, ncomponents);
      }
#endif
    }
  }
}
#endif

#ifdef USE_PARTIO
//! Write Partio files
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::write_partio(mpm::Index step, mpm::Index max_steps) {

  // MPI parallel partio file
  int mpi_rank = 0;
  int mpi_size = 1;
#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Get Partio file extensions
  const std::string extension = ".bgeo";
  const std::string attribute = "partio";
  // Create filename
  auto file =
      io_->output_file(attribute, extension, uuid_, step, max_steps).string();
  // Write partio file
  mpm::partio::write_particles(file, mesh_->particles_hdf5());
}
#endif  // USE_PARTIO

//! Output results
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::write_outputs(mpm::Index step) {
  if (step % this->output_steps_ == 0) {
    // HDF5 outputs
    this->write_hdf5(step, this->nsteps_);
#ifdef USE_VTK
    // VTK outputs
    this->write_vtk(step, this->nsteps_);
#endif
#ifdef USE_PARTIO
    // Partio outputs
    this->write_partio(step, this->nsteps_);
#endif
  }
}

//! Return if a mesh is isoparametric
template <unsigned Tdim>
bool mpm::MPMBase<Tdim>::is_isoparametric() {
  bool isoparametric = true;

  try {
    const auto mesh_props = io_->json_object("mesh");
    isoparametric = mesh_props.at("isoparametric").template get<bool>();
  } catch (std::exception& exception) {
    console_->warn(
        "{} #{}: Isoparametric status of mesh, using \"isoparametric\" "
        "as default; {}",
        __FILE__, __LINE__, exception.what());
    isoparametric = true;
  }
  return isoparametric;
}

//! Return if interface and levelset are active
template <unsigned Tdim>
bool mpm::MPMBase<Tdim>::is_levelset() {
  bool levelset_active = true;

  const auto mesh_props = io_->json_object("mesh");
  if (mesh_props.find("interface") != mesh_props.end()) {
    this->interface_ = true;
    this->interface_type_ =
        mesh_props["interface"]["interface_type"].template get<std::string>();
    if (interface_type_ != "levelset") levelset_active = false;
  } else
    levelset_active = false;

  return levelset_active;
}

//! Initialise loads
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_loads() {
  auto loads = io_->json_object("external_loading_conditions");
  // Initialise gravity loading
  gravity_.setZero();
  if (loads.at("gravity").is_array() &&
      loads.at("gravity").size() == gravity_.size()) {
    for (unsigned i = 0; i < gravity_.size(); ++i) {
      gravity_[i] = loads.at("gravity").at(i);
    }

    // Assign initial particle acceleration as gravity
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::assign_acceleration,
                  std::placeholders::_1, gravity_));
  } else {
    throw std::runtime_error(
        "mpm::base::initialise_loads(): Specified gravity dimension is "
        "invalid");
  }

  // Create a file reader
  const std::string io_type =
      io_->json_object("mesh")["io_type"].template get<std::string>();
  auto reader = Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

  // Read and assign particles surface tractions
  if (loads.find("particle_surface_traction") != loads.end()) {
    for (const auto& ptraction : loads["particle_surface_traction"]) {
      // Get the math function
      std::shared_ptr<FunctionBase> tfunction = nullptr;
      // If a math function is defined set to function or use scalar
      if (ptraction.find("math_function_id") != ptraction.end())
        tfunction = math_functions_.at(
            ptraction.at("math_function_id").template get<unsigned>());
      // Set id
      int pset_id = ptraction.at("pset_id").template get<int>();
      // Direction
      unsigned dir = ptraction.at("dir").template get<unsigned>();
      // Traction
      double traction = ptraction.at("traction").template get<double>();

      // Create particle surface tractions
      bool particles_tractions =
          mesh_->create_particles_tractions(tfunction, pset_id, dir, traction);
      if (!particles_tractions)
        throw std::runtime_error(
            "mpm::base::initialise_loads(): Particles tractions are not "
            "properly assigned");
    }
  } else
    console_->warn(
        "#{}: Particle surface tractions are undefined; Particle surface "
        "tractions JSON data not found",
        __LINE__);

  // Read and assign nodal concentrated forces
  if (loads.find("concentrated_nodal_forces") != loads.end()) {
    for (const auto& nforce : loads["concentrated_nodal_forces"]) {
      // Forces are specified in a file
      if (nforce.find("file") != nforce.end()) {
        std::string force_file = nforce.at("file").template get<std::string>();
        bool nodal_forces = mesh_->assign_nodal_concentrated_forces(
            reader->read_forces(io_->file_name(force_file)));
        if (!nodal_forces)
          throw std::runtime_error(
              "mpm::base::initialise_loads(): Nodal force file is invalid, "
              "forces are not properly assigned");
        set_node_concentrated_force_ = true;
      } else {
        // Get the math function
        std::shared_ptr<FunctionBase> ffunction = nullptr;
        if (nforce.find("math_function_id") != nforce.end())
          ffunction = math_functions_.at(
              nforce.at("math_function_id").template get<unsigned>());
        // Set id
        int nset_id = nforce.at("nset_id").template get<int>();
        // Direction
        unsigned dir = nforce.at("dir").template get<unsigned>();
        // Traction
        double force = nforce.at("force").template get<double>();

        // Read and assign nodal concentrated forces
        bool nodal_force = mesh_->assign_nodal_concentrated_forces(
            ffunction, nset_id, dir, force);
        if (!nodal_force)
          throw std::runtime_error(
              "mpm::base::initialise_loads(): Concentrated nodal forces are "
              "not properly assigned");
        set_node_concentrated_force_ = true;
      }
    }
  } else
    console_->warn(
        "#{}: Concentrated nodal forces are undefined; Concentrated nodal "
        "forces JSON data not found",
        __LINE__);
}

//! Initialise math functions
template <unsigned Tdim>
bool mpm::MPMBase<Tdim>::initialise_math_functions(const Json& math_functions) {
  bool status = true;
  try {
    // Get materials properties
    for (const auto& function_props : math_functions) {

      // Get math function id
      auto function_id = function_props["id"].template get<unsigned>();

      // Get function type
      const std::string function_type =
          function_props["type"].template get<std::string>();

      // Initiate another function_prop to be passed
      auto function_props_update = function_props;

      // Create a file reader
      const std::string io_type =
          io_->json_object("mesh")["io_type"].template get<std::string>();
      const auto& reader =
          Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

      // Math function is specified in a file, replace function_props_update
      if (function_props.find("file") != function_props.end()) {
        // Read file and store in array of vectors
        std::string math_file = io_->file_name(
            function_props.at("file").template get<std::string>());
        const auto& xfx_values = reader->read_math_functions(math_file);

        function_props_update["xvalues"] = xfx_values[0];
        function_props_update["fxvalues"] = xfx_values[1];
      }

      // Create a new function from JSON object
      auto function =
          Factory<mpm::FunctionBase, unsigned, const Json&>::instance()->create(
              function_type, std::move(function_id), function_props_update);

      // Add math function to list
      auto insert_status =
          math_functions_.insert(std::make_pair(function->id(), function));

      // If insert math function failed
      if (!insert_status.second) {
        status = false;
        throw std::runtime_error(
            "Invalid properties for new math function, fn insertion failed");
      }
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: Math functions are not properly specified; {}",
                    __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Node entity sets
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::node_entity_sets(const Json& mesh_props,
                                          bool check_duplicates) {
  try {
    if (mesh_props.find("entity_sets") != mesh_props.end()) {
      std::string entity_sets =
          mesh_props["entity_sets"].template get<std::string>();
      if (!io_->file_name(entity_sets).empty()) {
        bool node_sets = mesh_->create_node_sets(
            (io_->entity_sets(io_->file_name(entity_sets), "node_sets")),
            check_duplicates);
        if (!node_sets)
          throw std::runtime_error(
              "Node entity sets are not properly assigned");
      }
    } else
      throw std::runtime_error("Node entity set JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Node entity sets are undefined; {}", __LINE__,
                   exception.what());
  }
}

//! Node Euler angles
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::node_euler_angles(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("nodal_euler_angles") !=
            mesh_props["boundary_conditions"].end()) {
      std::string euler_angles =
          mesh_props["boundary_conditions"]["nodal_euler_angles"]
              .template get<std::string>();
      if (!io_->file_name(euler_angles).empty()) {
        bool rotation_matrices = mesh_->compute_nodal_rotation_matrices(
            mesh_io->read_euler_angles(io_->file_name(euler_angles)));
        if (!rotation_matrices)
          throw std::runtime_error(
              "Euler angles are not properly assigned/computed");
      }
    } else
      throw std::runtime_error("Euler angles JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Euler angles are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Nodal acceleration constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_acceleration_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign acceleration constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("acceleration_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over acceleration constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["acceleration_constraints"]) {
        // Acceleration constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string acceleration_constraints_file =
              constraints.at("file").template get<std::string>();
          bool acceleration_constraints =
              constraints_->assign_nodal_acceleration_constraints(
                  mesh_io->read_acceleration_constraints(
                      io_->file_name(acceleration_constraints_file)));
          if (!acceleration_constraints)
            throw std::runtime_error(
                "Acceleration constraints are not properly assigned");

        } else {
          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Direction
          unsigned dir = constraints.at("dir").template get<unsigned>();
          // Acceleration
          double acceleration =
              constraints.at("acceleration").template get<double>();
          // Get the math function
          std::shared_ptr<FunctionBase> afunction = nullptr;
          if (constraints.find("math_function_id") != constraints.end())
            afunction = math_functions_.at(
                constraints.at("math_function_id").template get<unsigned>());
          // Add acceleration constraint to mesh
          auto acceleration_constraint =
              std::make_shared<mpm::AccelerationConstraint>(nset_id, afunction,
                                                            dir, acceleration);
          mesh_->create_nodal_acceleration_constraint(nset_id,
                                                      acceleration_constraint);
          bool acceleration_constraints =
              constraints_->assign_nodal_acceleration_constraint(
                  nset_id, acceleration_constraint);
          if (!acceleration_constraints)
            throw std::runtime_error(
                "Nodal acceleration constraints are not properly assigned");
        }
      }
    } else
      throw std::runtime_error("Acceleration constraints JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Acceleration constraints are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Nodal velocity constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_velocity_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign velocity constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("velocity_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over velocity constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["velocity_constraints"]) {
        // Velocity constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string velocity_constraints_file =
              constraints.at("file").template get<std::string>();
          bool velocity_constraints =
              constraints_->assign_nodal_velocity_constraints(
                  mesh_io->read_velocity_constraints(
                      io_->file_name(velocity_constraints_file)));
          if (!velocity_constraints)
            throw std::runtime_error(
                "Velocity constraints are not properly assigned");

        } else {
          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Direction
          unsigned dir = constraints.at("dir").template get<unsigned>();
          // Velocity
          double velocity = constraints.at("velocity").template get<double>();
          // Add velocity constraint to mesh
          auto velocity_constraint =
              std::make_shared<mpm::VelocityConstraint>(nset_id, dir, velocity);
          bool velocity_constraints =
              constraints_->assign_nodal_velocity_constraint(
                  nset_id, velocity_constraint);
          if (!velocity_constraints)
            throw std::runtime_error(
                "Nodal velocity constraints are not properly assigned");
        }
      }
    } else
      throw std::runtime_error("Velocity constraints JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Velocity constraints are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Nodal displacement constraints for implicit solver
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_displacement_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign displacement constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("displacement_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over displacement constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["displacement_constraints"]) {
        // Displacement constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string displacement_constraints_file =
              constraints.at("file").template get<std::string>();
          bool displacement_constraints =
              constraints_->assign_nodal_displacement_constraints(
                  mesh_io->read_displacement_constraints(
                      io_->file_name(displacement_constraints_file)));
          if (!displacement_constraints)
            throw std::runtime_error(
                "Displacement constraints are not properly assigned");

        } else {
          // Get the math function
          std::shared_ptr<FunctionBase> dfunction = nullptr;
          if (constraints.find("math_function_id") != constraints.end())
            dfunction = math_functions_.at(
                constraints.at("math_function_id").template get<unsigned>());
          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Direction
          unsigned dir = constraints.at("dir").template get<unsigned>();
          // Displacement
          double displacement =
              constraints.at("displacement").template get<double>();
          // Add displacement constraint to mesh
          auto displacement_constraint =
              std::make_shared<mpm::DisplacementConstraint>(nset_id, dir,
                                                            displacement);
          bool displacement_constraints =
              constraints_->assign_nodal_displacement_constraint(
                  dfunction, nset_id, displacement_constraint);
          if (!displacement_constraints)
            throw std::runtime_error(
                "Nodal displacement constraints are not properly assigned");
        }
      }
    } else
      throw std::runtime_error("Displacement constraints JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Displacement constraints are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Nodal frictional constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_frictional_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign friction constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("friction_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over friction constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["friction_constraints"]) {
        // Friction constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string friction_constraints_file =
              constraints.at("file").template get<std::string>();
          bool friction_constraints =
              constraints_->assign_nodal_friction_constraints(
                  mesh_io->read_friction_constraints(
                      io_->file_name(friction_constraints_file)));
          if (!friction_constraints)
            throw std::runtime_error(
                "Friction constraints are not properly assigned");

        } else {

          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Direction (normal)
          unsigned dir = constraints.at("dir").template get<unsigned>();
          // Sign of normal direction
          int sign_n = constraints.at("sign_n").template get<int>();
          // Friction
          double friction = constraints.at("friction").template get<double>();
          // Add friction constraint to mesh
          auto friction_constraint = std::make_shared<mpm::FrictionConstraint>(
              nset_id, dir, sign_n, friction);
          bool friction_constraints =
              constraints_->assign_nodal_frictional_constraint(
                  nset_id, friction_constraint);
          if (!friction_constraints)
            throw std::runtime_error(
                "Nodal friction constraints are not properly assigned");
        }
      }
    } else
      throw std::runtime_error("Friction constraints JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Friction conditions are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Nodal adhesional constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_adhesional_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign adhesion constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("adhesion_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over adhesion constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["adhesion_constraints"]) {
        // Adhesion constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string adhesion_constraints_file =
              constraints.at("file").template get<std::string>();
          bool adhesion_constraints =
              constraints_->assign_nodal_adhesion_constraints(
                  mesh_io->read_adhesion_constraints(
                      io_->file_name(adhesion_constraints_file)));
          if (!adhesion_constraints)
            throw std::runtime_error(
                "Adhesion constraints are not properly assigned");

        } else {  // Entity sets

          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Direction (normal)
          unsigned dir = constraints.at("dir").template get<unsigned>();
          // Sign of normal direction
          int sign_n = constraints.at("sign_n").template get<int>();
          // Adhesion
          double adhesion = constraints.at("adhesion").template get<double>();
          // h_min
          double h_min = constraints.at("h_min").template get<double>();
          // nposition
          int nposition = constraints.at("nposition").template get<int>();
          // Add adhesion constraint to mesh
          auto adhesion_constraint = std::make_shared<mpm::AdhesionConstraint>(
              nset_id, dir, sign_n, adhesion, h_min, nposition);
          bool adhesion_constraints =
              constraints_->assign_nodal_adhesional_constraint(
                  nset_id, adhesion_constraint);
          if (!adhesion_constraints)
            throw std::runtime_error(
                "Nodal adhesion constraints are not properly assigned");
        }
      }
    } else
      throw std::runtime_error("Adhesion constraints JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Adhesion conditions are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Interface inputs (includes multimaterial and levelset)
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::interface_inputs(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign interface inputs
    if (mesh_props.find("interface") != mesh_props.end()) {
      this->interface_ = true;
      // Get interface type
      this->interface_type_ =
          mesh_props["interface"]["interface_type"].template get<std::string>();
      if (interface_type_ == "levelset") {
        // Check if levelset inputs are specified in a file
        if (mesh_props["interface"].find("location") !=
            mesh_props["interface"].end()) {
          // Retrieve the file name
          std::string levelset_input_file =
              mesh_props["interface"]["location"].template get<std::string>();
          // Retrieve levelset info from file
          bool levelset_inputs =
              mesh_->assign_nodal_levelset_values(mesh_io->read_levelset_input(
                  io_->file_name(levelset_input_file)));
          if (!levelset_inputs) {
            throw std::runtime_error(
                "Levelset interface is undefined; Levelset inputs are not "
                "properly assigned");
          }
        } else {
          throw std::runtime_error(
              "Levelset interface is undefined; Levelset location is not "
              "specified");
        }
        // Check if levelset damping factor is specified
        if (mesh_props["interface"].find("damping") !=
            mesh_props["interface"].end()) {
          // Retrieve levelset damping factor
          levelset_damping_ =
              mesh_props["interface"]["damping"].template get<double>();
          if ((levelset_damping_ < 0.) || (levelset_damping_ > 1.)) {
            levelset_damping_ = 0.05;
            throw std::runtime_error(
                "Levelset damping factor is not properly specified, using "
                "0.05 as default");
          }
        } else {
          throw std::runtime_error(
              "Levelset damping is not specified, using 0.05 as default");
        }
        // Check if levelset contact velocity update scheme is specified
        if (mesh_props["interface"].find("velocity_update") !=
            mesh_props["interface"].end()) {
          // Retrieve levelset damping factor
          std::string levelset_velocity_update_ =
              mesh_props["interface"]["velocity_update"]
                  .template get<std::string>();
          if (levelset_velocity_update_ == "global")
            levelset_pic_ = false;
          else if (levelset_velocity_update_ != "pic") {
            throw std::runtime_error(
                "Levelset contact velocity update is not properly specified, "
                " using \"pic\" as default");
          }
        } else {
          throw std::runtime_error(
              "Levelset contact velocity update is not specified, "
              " using \"pic\" as default");
        }
        // Check if levelset violation corrector is specified
        if (mesh_props["interface"].find("violation_corrector") !=
            mesh_props["interface"].end()) {
          // Retrieve levelset violation corrector
          levelset_violation_corrector_ =
              mesh_props["interface"]["violation_corrector"]
                  .template get<double>();
          if ((levelset_violation_corrector_ < 0.) ||
              (levelset_violation_corrector_ > 1.)) {
            levelset_violation_corrector_ = 0.01;
            throw std::runtime_error(
                "Levelset violation corrector is not properly specified, using "
                "0.01 as default");
          }
        } else {
          throw std::runtime_error(
              "Levelset violation corrector is not specified, using 0.01 as "
              "default");
        }
      } else if (interface_type_ == "multimaterial") {
        throw std::runtime_error(
            "Interfaces are undefined; Interface type \"multimaterial\" not "
            "supported");
      } else {
        throw std::runtime_error(
            "Interfaces are undefined; Interface type not properly specified");
      }
    } else {
      throw std::runtime_error(
          "Interfaces are undefined; Interfaces JSON data not found");
    }
  } catch (std::exception& exception) {
    console_->warn("#{}: {}", __LINE__, exception.what());
  }
}

// Nodal pressure constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_pressure_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign pressure constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("pressure_constraints") !=
            mesh_props["boundary_conditions"].end()) {

      // Iterate over pressure constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["pressure_constraints"]) {
        // Pore pressure constraint phase indice
        unsigned constraint_phase = constraints["phase_id"];

        // Pore pressure constraints are specified in a file
        if (constraints.find("file") != constraints.end()) {
          std::string pressure_constraints_file =
              constraints.at("file").template get<std::string>();
          bool ppressure_constraints =
              constraints_->assign_nodal_pressure_constraints(
                  constraint_phase,
                  mesh_io->read_pressure_constraints(
                      io_->file_name(pressure_constraints_file)));
          if (!ppressure_constraints)
            throw std::runtime_error(
                "Pore pressure constraints are not properly assigned");
        } else {
          // Get the math function
          std::shared_ptr<FunctionBase> pfunction = nullptr;
          if (constraints.find("math_function_id") != constraints.end())
            pfunction = math_functions_.at(
                constraints.at("math_function_id").template get<unsigned>());
          // Set id
          int nset_id = constraints.at("nset_id").template get<int>();
          // Pressure
          double pressure = constraints.at("pressure").template get<double>();
          // Add pressure constraint to mesh
          constraints_->assign_nodal_pressure_constraint(
              pfunction, nset_id, constraint_phase, pressure);
        }
      }
    } else
      throw std::runtime_error("Pressure constraints JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Nodal pressure constraints are undefined; {}",
                   __LINE__, exception.what());
  }
}

// Assign nodal absorbing constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_absorbing_constraints(
    const Json& mesh_props, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io) {
  try {
    // Read and assign absorbing constraints
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find("absorbing_constraints") !=
            mesh_props["boundary_conditions"].end()) {
      // Iterate over absorbing constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]["absorbing_constraints"]) {
        // Set id
        int nset_id = constraints.at("nset_id").template get<int>();
        // Direction
        unsigned dir = constraints.at("dir").template get<unsigned>();
        // Delta
        double delta = constraints.at("delta").template get<double>();
        // h_min
        double h_min = constraints.at("h_min").template get<double>();
        // a
        double a = constraints.at("a").template get<double>();
        // b
        double b = constraints.at("b").template get<double>();
        // position
        std::string position_str =
            constraints.at("position").template get<std::string>();
        mpm::Position position = mpm::Position::None;
        if (position_str == "corner")
          position = mpm::Position::Corner;
        else if (position_str == "edge")
          position = mpm::Position::Edge;
        else if (position_str == "face")
          position = mpm::Position::Face;
        // Add absorbing constraint to mesh
        auto absorbing_constraint = std::make_shared<mpm::AbsorbingConstraint>(
            nset_id, dir, delta, h_min, a, b, position);
        bool absorbing_constraints =
            constraints_->assign_nodal_absorbing_constraint(
                nset_id, absorbing_constraint);
        if (!absorbing_constraints)
          throw std::runtime_error(
              "Nodal absorbing constraints are not properly assigned");
        // Assign node set IDs and list of constraints
        constraints_->assign_absorbing_id_ptr(nset_id, absorbing_constraint);
        // Set bool for solve loop
        absorbing_boundary_ = true;
      }
    } else
      throw std::runtime_error("Absorbing constraints JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Absorbing conditions are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Apply nodal absorbing constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::nodal_absorbing_constraints() {
  std::vector<std::shared_ptr<mpm::AbsorbingConstraint>> absorbing_constraint =
      constraints_->absorbing_ptrs();
  std::vector<unsigned> absorbing_nset_id = constraints_->absorbing_ids();
  for (unsigned i = 0; i < absorbing_nset_id.size(); i++) {
    auto nset_id = absorbing_nset_id.at(i);
    const auto& a_constraint = absorbing_constraint.at(i);
    bool absorbing_constraints =
        constraints_->assign_nodal_absorbing_constraint(nset_id, a_constraint);
    if (!absorbing_constraints)
      throw std::runtime_error(
          "mpm::base::nodal_absorbing_constraints(): Nodal absorbing "
          "constraints are not properly applied");
  }
}

//! Cell entity sets
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::cell_entity_sets(const Json& mesh_props,
                                          bool check_duplicates) {
  try {
    if (mesh_props.find("entity_sets") != mesh_props.end()) {
      // Read and assign cell sets
      std::string entity_sets =
          mesh_props["entity_sets"].template get<std::string>();
      if (!io_->file_name(entity_sets).empty()) {
        bool cell_sets = mesh_->create_cell_sets(
            (io_->entity_sets(io_->file_name(entity_sets), "cell_sets")),
            check_duplicates);
        if (!cell_sets)
          throw std::runtime_error(
              "Cell entity sets are not properly assigned or "
              "JSON data not found");
      }
    } else
      throw std::runtime_error("Cell entity sets JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Cell entity sets are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Particles cells
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particles_cells(
    const Json& mesh_props,
    const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io) {
  try {
    if (mesh_props.find("particle_cells") != mesh_props.end()) {
      std::string fparticles_cells =
          mesh_props["particle_cells"].template get<std::string>();

      if (!io_->file_name(fparticles_cells).empty()) {
        bool particles_cells =
            mesh_->assign_particles_cells(particle_io->read_particles_cells(
                io_->file_name(fparticles_cells)));
        if (!particles_cells)
          throw std::runtime_error(
              "Particle cells are not properly assigned to particles");
      }
    } else
      throw std::runtime_error("Particle cells JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Particle cells are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Particles volumes
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particles_volumes(
    const Json& mesh_props,
    const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io) {
  try {
    if (mesh_props.find("particles_volumes") != mesh_props.end()) {
      std::string fparticles_volumes =
          mesh_props["particles_volumes"].template get<std::string>();
      if (!io_->file_name(fparticles_volumes).empty()) {
        bool particles_volumes =
            mesh_->assign_particles_volumes(particle_io->read_particles_volumes(
                io_->file_name(fparticles_volumes)));
        if (!particles_volumes)
          throw std::runtime_error(
              "Particles volumes are not properly assigned");
      }
    } else
      throw std::runtime_error("Particle volumes JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Particle volumes are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Particle velocity constraints
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particle_velocity_constraints() {
  auto mesh_props = io_->json_object("mesh");
  // Create a file reader
  const std::string io_type =
      io_->json_object("mesh")["io_type"].template get<std::string>();
  auto reader = Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

  try {
    if (mesh_props.find("boundary_conditions") != mesh_props.end() &&
        mesh_props["boundary_conditions"].find(
            "particles_velocity_constraints") !=
            mesh_props["boundary_conditions"].end()) {

      // Iterate over velocity constraints
      for (const auto& constraints :
           mesh_props["boundary_conditions"]
                     ["particles_velocity_constraints"]) {

        // Set id
        int pset_id = constraints.at("pset_id").template get<int>();
        // Direction
        unsigned dir = constraints.at("dir").template get<unsigned>();
        // Velocity
        double velocity = constraints.at("velocity").template get<double>();
        // Add velocity constraint to mesh
        auto velocity_constraint =
            std::make_shared<mpm::VelocityConstraint>(pset_id, dir, velocity);
        mesh_->create_particle_velocity_constraint(pset_id,
                                                   velocity_constraint);
      }
    } else
      throw std::runtime_error(
          "Particle velocity constraints JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Particle velocity constraints are undefined; {}",
                   __LINE__, exception.what());
  }
}
// Delete_particles function
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::delete_particles() {
  try {
    auto particles_treatment_prop = io_->particles_treatment();

    if (particles_treatment_prop.contains("delete_particle_ids")) {
      for (const auto& id : particles_treatment_prop.at("delete_particle_ids")) {
        mpm::Index pid = id.get<mpm::Index>();

        bool success = mesh_->remove_particle_by_id(pid);

      }
      
      console_->info("Successfully deleted particles.");
    }
  } catch (const std::exception& e) {
    std::cerr << "Error deleting particles: " << e.what() << "\n";
  }
}

// Reset_particles_displacement function
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::reset_particles_displacement() {
  try {
    auto particles_treatment_prop = io_->particles_treatment();

    if (particles_treatment_prop.contains("reset_displacement") &&
        particles_treatment_prop["reset_displacement"].get<bool>() == true) {

      // Iterate over particles and reset displacement
      mesh_->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Tdim>::reset_displacement,
          std::placeholders::_1));

      // Output success info
      console_->info("Successfully reset particle displacements.");
    }

  } catch (const std::exception& e) {
    std::cerr << "Error resetting particle displacements: " << e.what() << "\n";
  }
}

// Particles stresses
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particles_stresses(
    const Json& mesh_props,
    const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io) {
  try {
    if (mesh_props.find("particles_stresses") != mesh_props.end()) {
      // Get generator type
      const std::string type =
          mesh_props["particles_stresses"]["type"].template get<std::string>();
      if (type == "file") {
        std::string fparticles_stresses =
            mesh_props["particles_stresses"]["location"]
                .template get<std::string>();
        if (!io_->file_name(fparticles_stresses).empty()) {

          // Get stresses of all particles
          const auto all_particles_stresses =
              particle_io->read_particles_stresses(
                  io_->file_name(fparticles_stresses));

          // Read and assign particles stresses
          if (!mesh_->assign_particles_stresses(all_particles_stresses))
            throw std::runtime_error(
                "Particles stresses are not properly assigned");
        }
      } else if (type == "isotropic") {
        Eigen::Matrix<double, 6, 1> in_stress;
        in_stress.setZero();
        if (mesh_props["particles_stresses"]["values"].is_array() &&
            mesh_props["particles_stresses"]["values"].size() ==
                in_stress.size()) {
          for (unsigned i = 0; i < in_stress.size(); ++i) {
            in_stress[i] = mesh_props["particles_stresses"]["values"].at(i);
          }
          mesh_->iterate_over_particles(
              std::bind(&mpm::ParticleBase<Tdim>::initial_stress,
                        std::placeholders::_1, in_stress));
        } else {
          throw std::runtime_error("Initial stress dimension is invalid");
        }
      }
    } else
      throw std::runtime_error("Particle stresses JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Particle stresses are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Particles pore pressures
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particles_pore_pressures(
    const Json& mesh_props,
    const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io) {
  try {
    if (mesh_props.find("particles_pore_pressures") != mesh_props.end()) {
      // Get generator type
      const std::string type = mesh_props["particles_pore_pressures"]["type"]
                                   .template get<std::string>();
      // Assign initial pore pressure by file
      if (type == "file") {
        std::string fparticles_pore_pressures =
            mesh_props["particles_pore_pressures"]["location"]
                .template get<std::string>();
        if (!io_->file_name(fparticles_pore_pressures).empty()) {
          // Read and assign particles pore pressures
          if (!mesh_->assign_particles_pore_pressures(
                  particle_io->read_scalar_properties(
                      io_->file_name(fparticles_pore_pressures))))
            throw std::runtime_error(
                "Particles pore pressures are not properly assigned");
        } else
          throw std::runtime_error(
              "Particle pore pressures JSON data not found");
      } else if (type == "water_table") {
        // Initialise water tables
        std::map<double, double> reference_points;
        // Vertical direction
        const unsigned dir_v = mesh_props["particles_pore_pressures"]["dir_v"]
                                   .template get<unsigned>();
        // Horizontal direction
        const unsigned dir_h = mesh_props["particles_pore_pressures"]["dir_h"]
                                   .template get<unsigned>();
        // Iterate over water tables
        for (const auto& water_table :
             mesh_props["particles_pore_pressures"]["water_tables"]) {
          // Position coordinate
          double position = water_table.at("position").template get<double>();
          // Direction
          double h0 = water_table.at("h0").template get<double>();
          // Add reference points to mesh
          reference_points.insert(std::make_pair<double, double>(
              static_cast<double>(position), static_cast<double>(h0)));
        }

        // Read gravity
        Eigen::Matrix<double, Tdim, 1> gravity =
            Eigen::Matrix<double, Tdim, 1>::Zero();
        auto loads = io_->json_object("external_loading_conditions");
        if (loads.contains("gravity")) {
          if (loads.at("gravity").is_array() &&
              loads.at("gravity").size() == gravity.size()) {
            for (unsigned i = 0; i < gravity.size(); ++i) {
              gravity[i] = loads.at("gravity").at(i);
            }
          } else {
            throw std::runtime_error("Specified gravity dimension is invalid");
          }
        } else {
          throw std::runtime_error(
              "In order to use the option water table, \"gravity\" should be "
              "specified in the \"external_loading_conditions\"");
        }

        // Initialise particles pore pressures by watertable
        mesh_->iterate_over_particles(std::bind(
            &mpm::ParticleBase<Tdim>::initialise_pore_pressure_watertable,
            std::placeholders::_1, dir_v, dir_h, gravity, reference_points));
      } else if (type == "isotropic") {
        const double pore_pressure =
            mesh_props["particles_pore_pressures"]["values"]
                .template get<double>();
        mesh_->iterate_over_particles(std::bind(
            &mpm::ParticleBase<Tdim>::assign_pressure, std::placeholders::_1,
            pore_pressure, mpm::ParticlePhase::Liquid));
      } else
        throw std::runtime_error(
            "Particle pore pressures generator type is not properly "
            "specified");
    } else
      throw std::runtime_error("Particle pore pressure JSON data not found");

  } catch (std::exception& exception) {
    console_->warn("#{}: Particle pore pressures are undefined; {}", __LINE__,
                   exception.what());
  }
}

//! Particle entity sets
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::particle_entity_sets(bool check_duplicates) {
  // Get mesh properties
  auto mesh_props = io_->json_object("mesh");
  // Read and assign particle sets
  try {
    if (mesh_props.find("entity_sets") != mesh_props.end()) {
      std::string entity_sets =
          mesh_props["entity_sets"].template get<std::string>();
      if (!io_->file_name(entity_sets).empty()) {
        bool particle_sets = mesh_->create_particle_sets(
            (io_->entity_sets(io_->file_name(entity_sets), "particle_sets")),
            check_duplicates);
        if (!particle_sets)
          throw std::runtime_error(
              "Particle entity sets are not properly assigned or "
              "JSON data not found");
      }
    } else
      throw std::runtime_error("Particle entity set JSON data not found");
  } catch (std::exception& exception) {
    console_->warn("#{}: Particle entity sets are undefined; {}", __LINE__,
                   exception.what());
  }
}

// Initialise Damping
template <unsigned Tdim>
bool mpm::MPMBase<Tdim>::initialise_damping(const Json& damping_props) {

  // Read damping JSON object
  bool status = true;
  try {
    // Read damping type
    std::string type = damping_props.at("type").template get<std::string>();
    if (type == "Cundall") damping_type_ = mpm::Damping::Cundall;

    // Read damping factor
    damping_factor_ = damping_props.at("damping_factor").template get<double>();

  } catch (std::exception& exception) {
    console_->warn("#{}: Damping parameters are not properly specified; {}",
                   __LINE__, exception.what());
    status = false;
  }

  return status;
}

//! Domain decomposition
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::mpi_domain_decompose(bool initial_step) {
#ifdef USE_MPI
  // Initialise MPI rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_size > 1 && mesh_->ncells() > 1) {

    // Initialize MPI
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    auto mpi_domain_begin = std::chrono::steady_clock::now();
    console_->info("Rank {}, Domain decomposition started\n", mpi_rank);

#ifdef USE_GRAPH_PARTITIONING
    // Create graph object if empty
    if (initial_step || graph_ == nullptr)
      graph_ = std::make_shared<Graph<Tdim>>(mesh_->cells());

    // Find number of particles in each cell across MPI ranks
    mesh_->find_nglobal_particles_cells();

    // Construct a weighted DAG
    graph_->construct_graph(mpi_size, mpi_rank);

    // Graph partitioning mode
    int mode = 4;  // FAST
    // Create graph partition
    graph_->create_partitions(&comm, mode);
    // Collect the partitions
    auto exchange_cells = graph_->collect_partitions(mpi_size, mpi_rank, &comm);

    // Identify shared nodes across MPI domains
    mesh_->find_domain_shared_nodes();
    // Identify ghost boundary cells
    mesh_->find_ghost_boundary_cells();

    // Delete all the particles which is not in local task parititon
    if (initial_step) mesh_->remove_all_nonrank_particles();
    // Transfer non-rank particles to appropriate cells
    else
      mesh_->transfer_nonrank_particles(exchange_cells);

#endif
    auto mpi_domain_end = std::chrono::steady_clock::now();
    console_->info("Rank {}, Domain decomposition: {} ms", mpi_rank,
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       mpi_domain_end - mpi_domain_begin)
                       .count());
  }
#endif  // MPI
}

//! MPM pressure smoothing
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::pressure_smoothing(unsigned phase) {
  // Assign pressure to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_pressure_to_nodes,
                std::placeholders::_1, phase));

  // Apply pressure constraint
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::apply_pressure_constraint,
                std::placeholders::_1, phase, this->dt_, this->step_),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

#ifdef USE_MPI
  int mpi_size = 1;

  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Run if there is more than a single MPI task
  if (mpi_size > 1) {
    // MPI all reduce nodal pressure
    mesh_->template nodal_halo_exchange<double, 1>(
        std::bind(&mpm::NodeBase<Tdim>::pressure, std::placeholders::_1, phase),
        std::bind(&mpm::NodeBase<Tdim>::assign_pressure, std::placeholders::_1,
                  phase, std::placeholders::_2));
  }
#endif

  // Smooth pressure over particles
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_pressure_smoothing,
                std::placeholders::_1, phase));
}

//! MPM implicit solver initialization
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_linear_solver(
    const Json& lin_solver_props,
    tsl::robin_map<
        std::string,
        std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>&
        linear_solver) {
  // Iterate over specific solver settings
  for (const auto& solver : lin_solver_props) {
    std::string dof = solver["dof"].template get<std::string>();
    std::string solver_type = solver["solver_type"].template get<std::string>();
    // NOTE: Only KrylovPETSC solver is supported for MPI
#ifdef USE_MPI
    // Get number of MPI ranks
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (solver_type != "KrylovPETSC" && mpi_size > 1) {
      console_->warn(
          "The linear solver for DOF \"{}\" in MPI setting is "
          "automatically set to default, \"KrylovPETSC\"; Only "
          "\"KrylovPETSC\" solver is supported for MPI",
          dof);
      solver_type = "KrylovPETSC";
    }
#endif
    unsigned max_iter = solver["max_iter"].template get<unsigned>();
    double tolerance = solver["tolerance"].template get<double>();
    auto lin_solver =
        Factory<mpm::SolverBase<Eigen::SparseMatrix<double>>, unsigned,
                double>::instance()
            ->create(solver_type, std::move(max_iter), std::move(tolerance));

    // Specific settings
    if (solver.contains("sub_solver_type"))
      lin_solver->set_sub_solver_type(
          solver["sub_solver_type"].template get<std::string>());
    if (solver.contains("preconditioner_type"))
      lin_solver->set_preconditioner_type(
          solver["preconditioner_type"].template get<std::string>());
    if (solver.contains("abs_tolerance"))
      lin_solver->set_abs_tolerance(
          solver["abs_tolerance"].template get<double>());
    if (solver.contains("div_tolerance"))
      lin_solver->set_div_tolerance(
          solver["div_tolerance"].template get<double>());
    if (solver.contains("verbosity"))
      lin_solver->set_verbosity(solver["verbosity"].template get<unsigned>());

    // Add solver set to map
    linear_solver.insert(
        std::pair<
            std::string,
            std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>(
            dof, lin_solver));
  }
}

//! Initialise nonlocal mesh
template <unsigned Tdim>
void mpm::MPMBase<Tdim>::initialise_nonlocal_mesh(const Json& mesh_props) {
  //! Shape function name
  const auto cell_type = mesh_props["cell_type"].template get<std::string>();
  try {
    // Initialise additional properties
    tsl::robin_map<std::string, double> nonlocal_properties;

    // Parameters for B-Spline elements
    if (cell_type.back() == 'B') {
      // Cell and node neighbourhood for quadratic B-Spline
      cell_neighbourhood_ = 1;
      node_neighbourhood_ = 3;

      // Initialise nonlocal node
      mesh_->iterate_over_nodes(
          std::bind(&mpm::NodeBase<Tdim>::initialise_nonlocal_node,
                    std::placeholders::_1));

      //! Read nodal type from entity sets
      if (mesh_props.find("nonlocal_mesh_properties") != mesh_props.end()) {
        const auto sf_type = mesh_props["nonlocal_mesh_properties"]["type"]
                                 .template get<std::string>();
        assert(sf_type == "BSPLINE");

        // Apply kernel correction
        bool kernel_correction = true;
        if (mesh_props["nonlocal_mesh_properties"].contains(
                "kernel_correction")) {
          kernel_correction =
              mesh_props["nonlocal_mesh_properties"]["kernel_correction"]
                  .template get<bool>();
        }
        nonlocal_properties.insert(std::pair<std::string, bool>(
            "kernel_correction", kernel_correction));

        // Iterate over node type
        for (const auto& node_type :
             mesh_props["nonlocal_mesh_properties"]["node_types"]) {
          // Set id
          int nset_id = node_type.at("nset_id").template get<int>();
          // Direction
          unsigned dir = node_type.at("dir").template get<unsigned>();
          // Type
          unsigned type = node_type.at("type").template get<unsigned>();
          // Assign nodal nonlocal type
          mesh_->assign_nodal_nonlocal_type(nset_id, dir, type);
        }
      }
    }
    // Parameters for LME elements
    else if (cell_type.back() == 'L') {
      //! Read nodal type from entity sets
      if (mesh_props.find("nonlocal_mesh_properties") != mesh_props.end()) {
        const auto sf_type = mesh_props["nonlocal_mesh_properties"]["type"]
                                 .template get<std::string>();
        assert(sf_type == "LME");

        // Gamma parameter
        const double gamma = mesh_props["nonlocal_mesh_properties"]["gamma"]
                                 .template get<double>();

        // Support tolerance
        double tol0 = 1.e-6;
        if (mesh_props["nonlocal_mesh_properties"].contains(
                "support_tolerance"))
          tol0 = mesh_props["nonlocal_mesh_properties"]["support_tolerance"]
                     .template get<double>();

        // Average mesh size
        double h;
        if (mesh_props["nonlocal_mesh_properties"].contains("mesh_size"))
          h = mesh_props["nonlocal_mesh_properties"]["mesh_size"]
                  .template get<double>();
        else
          h = mesh_->compute_average_cell_size();

        // Anisotropy parameter
        bool anisotropy = false;
        if (mesh_props["nonlocal_mesh_properties"].contains("anisotropy")) {
          anisotropy = mesh_props["nonlocal_mesh_properties"]["anisotropy"]
                           .template get<bool>();
        }
        nonlocal_properties.insert(
            std::pair<std::string, bool>("anisotropy", anisotropy));

        // Calculate beta
        const double beta = gamma / (h * h);
        nonlocal_properties.insert(
            std::pair<std::string, double>("beta", beta));

        // Calculate support radius automatically
        const double r = std::sqrt(-std::log(tol0) / gamma) * h;
        nonlocal_properties.insert(
            std::pair<std::string, double>("support_radius", r));

        // Cell and node neighbourhood for LME
        cell_neighbourhood_ = static_cast<unsigned>(floor(r / h));
        node_neighbourhood_ = 1 + 2 * cell_neighbourhood_;
      }
    } else {
      throw std::runtime_error(
          "Unable to initialise nonlocal mesh for cell type \"" + cell_type +
          "\"");
    }

    //! Update number of nodes in cell
    mesh_->upgrade_cells_to_nonlocal(cell_type, cell_neighbourhood_,
                                     nonlocal_properties);

  } catch (std::exception& exception) {
    console_->warn("{} #{}: initialising nonlocal mesh failed; {}", __FILE__,
                   __LINE__, exception.what());
  }
}