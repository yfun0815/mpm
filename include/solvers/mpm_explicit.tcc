//! Constructor
template <unsigned Tdim>
mpm::MPMExplicit<Tdim>::MPMExplicit(const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  //! Logger
  console_ = spdlog::get("MPMExplicit");
  //! Stress update
  if (this->stress_update_ == "usl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSL<Tdim>>(mesh_, dt_);
  else if (this->stress_update_ == "musl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeMUSL<Tdim>>(mesh_, dt_);
  else
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSF<Tdim>>(mesh_, dt_);

  if (this->interface_)
    if (this->interface_type_ == "multimaterial")
      contact_ = std::make_shared<mpm::ContactFriction<Tdim>>(mesh_);
    else if (this->interface_type_ == "levelset")
      contact_ = std::make_shared<mpm::ContactLevelset<Tdim>>(mesh_);
    else  // default is "none"
      contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
  else
    contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
}

//! MPM Explicit solver
template <unsigned Tdim>
bool mpm::MPMExplicit<Tdim>::solve() {
  bool status = true;

  console_->info("MPM analysis type {}", io_->analysis_type());

  // Initialise MPI rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Phase
  const unsigned phase = 0;

  // Test if checkpoint resume is needed
  bool resume = false;
  if (analysis_.find("resume") != analysis_.end())
    resume = analysis_["resume"]["resume"].template get<bool>();

  // Enable repartitioning if resume is done with particles generated outside
  // the MPM code.
  bool repartition = false;
  if (analysis_.find("resume") != analysis_.end() &&
      analysis_["resume"].find("repartition") != analysis_["resume"].end())
    repartition = analysis_["resume"]["repartition"].template get<bool>();

  // Pressure smoothing
  pressure_smoothing_ = io_->analysis_bool("pressure_smoothing");

  // Initialise material
  this->initialise_materials();

  // Initialise mesh
  this->initialise_mesh();

  // Check point resume
  if (resume) {
    bool check_resume = this->checkpoint_resume();
    if (!check_resume) resume = false;
  }

  // Resume or Initialise
  bool initial_step = (resume == true) ? false : true;
  if (resume) {
    if (repartition) {
      this->mpi_domain_decompose(initial_step);
    } else {
      mesh_->resume_domain_cell_ranks();
#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
      MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    }
    // delete the provided particles by id
    this->delete_particles();
    // reset all particles displacement
    this->reset_particles_displacement();
    //! Particle entity sets and velocity constraints
    this->particle_entity_sets(false);
    this->particle_velocity_constraints();
  } else {
    // Initialise particles
    this->initialise_particles();
    // delete the provided particles by id
    this->delete_particles();
    // Compute mass
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  
  // Create nodal properties
  if (interface_ or absorbing_boundary_) mesh_->create_nodal_properties();

  // Initialise levelset properties
  if (this->interface_type_ == "levelset")
    contact_->initialise_levelset_properties(
        this->levelset_damping_, this->levelset_pic_,
        this->levelset_violation_corrector_);

  // Initialise loading conditions
  this->initialise_loads();

  // Write initial outputs
  if (!resume) this->write_outputs(this->step_);

  auto solver_begin = std::chrono::steady_clock::now();
  // Main loop
  int last_percent = -1;

  for (; step_ < nsteps_; ++step_) {
      if (mpi_rank == 0) {
          int percent = static_cast<int>(100.0 * step_ / nsteps_);
          if (percent != last_percent) {
              last_percent = percent;
              int bar_width = 50;
              int pos = percent * bar_width / 100;

              std::string bar = "[";
              for (int i = 0; i < bar_width; ++i) {
                  if (i < pos) bar += "#";
                  else if (i == pos) bar += ">";
                  else bar += "-";
              }
              bar += "]";

              console_->info("Step: {} of {}. {}% {}\n", step_, nsteps_, percent, bar);
          }
      }

#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    // Run load balancer at a specified frequency
    if (step_ % nload_balance_steps_ == 0 && step_ != 0)
      this->mpi_domain_decompose(false);
#endif
#endif

    // Inject particles
    mesh_->inject_particles(step_ * dt_);

    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise();

    // Initialise multimaterial nodal properties
    contact_->initialise();  // multimaterial interface

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(velocity_update_, phase);

    // Contact forces at nodes
    contact_->compute_contact_forces(dt_);  // levelset interface
    contact_->compute_contact_forces();     // multimaterial interface

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Compute forces
    mpm_scheme_->compute_forces(gravity_, phase, step_,
                                set_node_concentrated_force_);

    // Apply Absorbing Constraint
    if (absorbing_boundary_) {
      mpm_scheme_->absorbing_boundary_properties();
      this->nodal_absorbing_constraints();
    }

    // Particle kinematics
    mpm_scheme_->compute_particle_kinematics(velocity_update_, blending_ratio_,
                                             phase, "Cundall", damping_factor_,
                                             step_);

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->postcompute_nodal_kinematics(velocity_update_, phase);

    // Update Stress Last
    mpm_scheme_->postcompute_stress_strain(phase, pressure_smoothing_);

    // Locate particles
    mpm_scheme_->locate_particles(this->locate_particles_);

#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    mesh_->transfer_halo_particles();
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

    // Write outputs
    this->write_outputs(this->step_ + 1);
  }
  auto solver_end = std::chrono::steady_clock::now();
  console_->info("Rank {}, Explicit {} solver duration: {} ms", mpi_rank,
                 mpm_scheme_->scheme(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     solver_end - solver_begin)
                     .count());

  return status;
}
