//! Return coordinates of nodes in a mesh from input file
template <unsigned Tdim>
std::vector<Eigen::Matrix<double, Tdim, 1>>
    mpm::IOMeshAscii<Tdim>::read_mesh_nodes(const std::string& mesh) {
  // Nodal coordinates
  std::vector<VectorDim> coordinates;
  coordinates.clear();

  // input file stream
  std::fstream file;
  file.open(mesh.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      // bool to check firstline
      bool read_first_line = false;
      // Number of coordinate lines
      unsigned nlines = 0;
      // Coordinates
      Eigen::Matrix<double, Tdim, 1> coords;
      // # of nodes and cells
      unsigned nnodes = 0, ncells = 0;
      // ignore stream
      double ignore;

      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          while (istream.good()) {
            if (!read_first_line) {
              // Read number of nodes and cells
              istream >> nnodes >> ncells;
              coordinates.reserve(nnodes);
              read_first_line = true;
              break;
            }
            // Read until nodal information is present
            if (nlines <= nnodes) {
              // Read to coordinates
              for (unsigned i = 0; i < Tdim; ++i) istream >> coords[i];
              coordinates.emplace_back(coords);
              break;

            } else {
              // Ignore stream
              istream >> ignore;
            }
          }
          ++nlines;
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read mesh nodes: {}", exception.what());
    file.close();
  }

  return coordinates;
}

//! Return indices of nodes of cells in a mesh from input file
template <unsigned Tdim>
std::vector<std::vector<mpm::Index>> mpm::IOMeshAscii<Tdim>::read_mesh_cells(
    const std::string& mesh) {
  // Indices of nodes
  std::vector<std::vector<mpm::Index>> cells;
  cells.clear();

  std::fstream file;
  file.open(mesh.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      std::string line;
      // bool to check firstline
      bool read_first_line = false;
      // Number of coordinate lines
      unsigned nlines = 0;
      // Coordinates
      Eigen::Matrix<double, Tdim, 1> coords;
      // # of nodes and cells
      mpm::Index nnodes = 0, ncells = 0;
      // ignore stream
      double ignore;

      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // Vector of node ids for a cell
        std::vector<mpm::Index> nodes;
        nodes.clear();
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          while (istream.good()) {
            if (!read_first_line) {
              // Read number of nodes and cells
              istream >> nnodes >> ncells;
              cells.reserve(ncells);
              read_first_line = true;
              break;
            }
            // Ignore nodal coordinates
            if (nlines > nnodes) {
              // Read node ids of each cell
              mpm::Index nid;
              istream >> nid;
              nodes.emplace_back(nid);
            } else {
              // Ignore stream not related to node ids of cells
              istream >> ignore;
            }
          }
          ++nlines;
          // Check if nodes is not empty, before adding to cell
          if (!nodes.empty()) {
            cells.emplace_back(nodes);
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read mesh cells: {}", exception.what());
    file.close();
  }

  return cells;
}

//! Return coordinates of particles
template <unsigned Tdim>
std::vector<Eigen::Matrix<double, Tdim, 1>>
    mpm::IOMeshAscii<Tdim>::read_particles(const std::string& particles_file) {

  // Nodal coordinates
  std::vector<VectorDim> coordinates;
  coordinates.clear();

  // Expected number of particles
  mpm::Index nparticles;

  // bool to check firstline
  bool read_first_line = false;

  // input file stream
  std::fstream file;
  file.open(particles_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          while (istream.good()) {
            if (!read_first_line) {
              // Read number of nodes and cells
              istream >> nparticles;
              coordinates.reserve(nparticles);
              read_first_line = true;
              break;
            }
            // Coordinates
            Eigen::Matrix<double, Tdim, 1> coords;
            // Read to coordinates
            for (unsigned i = 0; i < Tdim; ++i) istream >> coords[i];
            coordinates.emplace_back(coords);
            break;
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read particle coordinates: {}", exception.what());
    file.close();
  }

  return coordinates;
}

//! Return stresses of particles
template <unsigned Tdim>
std::vector<Eigen::Matrix<double, 6, 1>>
    mpm::IOMeshAscii<Tdim>::read_particles_stresses(
        const std::string& particles_stresses) {

  // Nodal stresses
  std::vector<Eigen::Matrix<double, 6, 1>> stresses;
  stresses.clear();

  // Expected number of particles
  mpm::Index nparticles;

  // bool to check firstline
  bool read_first_line = false;

  // input file stream
  std::fstream file;
  file.open(particles_stresses.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          while (istream.good()) {
            if (!read_first_line) {
              // Read number of nodes and cells
              istream >> nparticles;
              stresses.reserve(nparticles);
              read_first_line = true;
              break;
            }
            // Stresses
            Eigen::Matrix<double, 6, 1> stress;
            // Read to stress
            for (unsigned i = 0; i < stress.size(); ++i) istream >> stress[i];
            stresses.emplace_back(stress);
            break;
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read particle stresses: {}", exception.what());
    file.close();
  }
  return stresses;
}

//! Return scalar properties for particles or nodes
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, double>>
    mpm::IOMeshAscii<Tdim>::read_scalar_properties(
        const std::string& scalar_file) {

  // Scalar properties for particles or nodes
  std::vector<std::tuple<mpm::Index, double>> scalar_properties;

  // input file stream
  std::fstream file;
  file.open(scalar_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Scalar
          double scalar;
          while (istream.good()) {
            // Read stream
            istream >> id >> scalar;
            scalar_properties.emplace_back(std::make_tuple(id, scalar));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read particle/node {} #{}: {}\n", __FILE__, __LINE__,
                    exception.what());
    file.close();
  }
  return scalar_properties;
}

//! Read pressure constraints file
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, double>>
    mpm::IOMeshAscii<Tdim>::read_pressure_constraints(
        const std::string& pressure_constraints_file) {
  // Particle pressure constraints
  std::vector<std::tuple<mpm::Index, double>> constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(pressure_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Pressure
          double pressure;
          while (istream.good()) {
            // Read stream
            istream >> id >> pressure;
            constraints.emplace_back(std::make_tuple(id, pressure));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read pressure constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return euler angles of nodes
template <unsigned Tdim>
std::map<mpm::Index, Eigen::Matrix<double, Tdim, 1>>
    mpm::IOMeshAscii<Tdim>::read_euler_angles(
        const std::string& nodal_euler_angles_file) {

  // Nodal euler angles
  std::map<mpm::Index, Eigen::Matrix<double, Tdim, 1>> euler_angles;
  euler_angles.clear();

  // input file stream
  std::fstream file;
  file.open(nodal_euler_angles_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Angles
          Eigen::Matrix<double, Tdim, 1> angles;
          while (istream.good()) {
            istream >> id;
            for (unsigned i = 0; i < Tdim; ++i) istream >> angles[i];
            euler_angles.emplace(std::make_pair(id, angles));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read euler angles: {}", exception.what());
    file.close();
  }
  return euler_angles;
}

//! Return particles volume
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, double>>
    mpm::IOMeshAscii<Tdim>::read_particles_volumes(
        const std::string& volume_file) {

  // particle volumes
  std::vector<std::tuple<mpm::Index, double>> volumes;
  volumes.clear();

  // input file stream
  std::fstream file;
  file.open(volume_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Volume
          double volume;
          while (istream.good()) {
            // Read stream
            istream >> id >> volume;
            volumes.emplace_back(std::make_tuple(id, volume));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read volume : {}", exception.what());
    file.close();
  }
  return volumes;
}

//! Return particles and their cells
template <unsigned Tdim>
std::vector<std::array<mpm::Index, 2>>
    mpm::IOMeshAscii<Tdim>::read_particles_cells(
        const std::string& particles_cells_file) {

  // Particle cells
  std::vector<std::array<mpm::Index, 2>> particles_cells;
  particles_cells.clear();

  // input file stream
  std::fstream file;
  file.open(particles_cells_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index pid, cid;
          while (istream.good()) {
            // Read stream
            istream >> pid >> cid;
            particles_cells.emplace_back(std::array<mpm::Index, 2>({pid, cid}));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read particles cells: {}", exception.what());
    file.close();
  }
  return particles_cells;
}

//! Write particles and their cells
template <unsigned Tdim>
void mpm::IOMeshAscii<Tdim>::write_particles_cells(
    const std::string& particles_cells_file,
    const std::vector<std::array<mpm::Index, 2>>& particles_cells) {

  // output file stream
  std::fstream file;
  file.open(particles_cells_file.c_str(), std::ios::out);

  for (const auto& particle_cell : particles_cells)
    file << particle_cell[0] << "\t" << particle_cell[1] << "\n";

  file.close();
}

//! Return velocity constraints of nodes or particles
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, double>>
    mpm::IOMeshAscii<Tdim>::read_velocity_constraints(
        const std::string& velocity_constraints_file) {

  // Nodal or particle velocity constraints
  std::vector<std::tuple<mpm::Index, unsigned, double>> constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(velocity_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction
          unsigned dir;
          // Velocity
          double velocity;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> velocity;
            constraints.emplace_back(std::make_tuple(id, dir, velocity));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read velocity constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return acceleration constraints of nodes or particles
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, double>>
    mpm::IOMeshAscii<Tdim>::read_acceleration_constraints(
        const std::string& acceleration_constraints_file) {

  // Nodal or particle acceleration constraints
  std::vector<std::tuple<mpm::Index, unsigned, double>> constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(acceleration_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction
          unsigned dir;
          // Velocity
          double acceleration;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> acceleration;
            constraints.emplace_back(std::make_tuple(id, dir, acceleration));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read acceleration constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return displacement constraints of nodes or particles
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, double>>
    mpm::IOMeshAscii<Tdim>::read_displacement_constraints(
        const std::string& displacement_constraints_file) {

  // Nodal or particle displacement constraints
  std::vector<std::tuple<mpm::Index, unsigned, double>> constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(displacement_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction
          unsigned dir;
          // Displacement
          double displacement;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> displacement;
            constraints.emplace_back(std::make_tuple(id, dir, displacement));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read displacement constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return friction constraints
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, int, double>>
    mpm::IOMeshAscii<Tdim>::read_friction_constraints(
        const std::string& friction_constraints_file) {

  // Nodal friction constraints
  std::vector<std::tuple<mpm::Index, unsigned, int, double>> constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(friction_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction (normal)
          unsigned dir;
          // Sign of normal direction
          int sign_n;
          // Friction
          double friction;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> sign_n >> friction;
            constraints.emplace_back(
                std::make_tuple(id, dir, sign_n, friction));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read friction constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return adhesion constraints of particles
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, int, double, double, int>>
    mpm::IOMeshAscii<Tdim>::read_adhesion_constraints(
        const std::string& adhesion_constraints_file) {

  // Nodal adhesion constraints
  std::vector<std::tuple<mpm::Index, unsigned, int, double, double, int>>
      constraints;
  constraints.clear();

  // input file stream
  std::fstream file;
  file.open(adhesion_constraints_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction (normal)
          unsigned dir;
          // Sign of normal direction
          int sign_n;
          // Adhesion
          double adhesion;
          // Cell height
          double h_min;
          // Node nposition
          int nposition = 0;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> sign_n >> adhesion >> h_min >> nposition;
            constraints.emplace_back(
                std::make_tuple(id, dir, sign_n, adhesion, h_min, nposition));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read adhesion constraints: {}", exception.what());
    file.close();
  }
  return constraints;
}

//! Return nodal levelset information
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, double, double, double, double>>
    mpm::IOMeshAscii<Tdim>::read_levelset_input(
        const std::string& levelset_input_file) {

  // Nodal levelset information
  std::vector<std::tuple<mpm::Index, double, double, double, double>>
      levelset_inputs;
  levelset_inputs.clear();

  // input file stream
  std::fstream file;
  file.open(levelset_input_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Levelset value
          double levelset;
          // Friction
          double levelset_mu;
          // Adhesion coefficient
          double levelset_alpha;
          // Barrier stiffness
          double barrier_stiffness;
          while (istream.good()) {
            // Read stream
            istream >> id >> levelset >> levelset_mu >> levelset_alpha >>
                barrier_stiffness;
            levelset_inputs.emplace_back(std::make_tuple(
                id, levelset, levelset_mu, levelset_alpha, barrier_stiffness));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read levelset inputs: {}", exception.what());
    file.close();
  }
  return levelset_inputs;
}

//! Return particles force
template <unsigned Tdim>
std::vector<std::tuple<mpm::Index, unsigned, double>>
    mpm::IOMeshAscii<Tdim>::read_forces(const std::string& force_file) {

  // particle forces
  std::vector<std::tuple<mpm::Index, unsigned, double>> forces;
  forces.clear();

  // input file stream
  std::fstream file;
  file.open(force_file.c_str(), std::ios::in);

  try {
    if (file.is_open() && file.good()) {
      // Line
      std::string line;
      while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        std::istringstream istream(line);
        // ignore comment lines (# or !) or blank lines
        if ((line.find('#') == std::string::npos) &&
            (line.find('!') == std::string::npos) && (line != "")) {
          // ID
          mpm::Index id;
          // Direction
          unsigned dir;
          // Force
          double force;
          while (istream.good()) {
            // Read stream
            istream >> id >> dir >> force;
            forces.emplace_back(std::make_tuple(id, dir, force));
          }
        }
      }
    } else {
      throw std::runtime_error("File not open or not good!");
    }
    file.close();
  } catch (std::exception& exception) {
    console_->error("Read force : {}", exception.what());
    file.close();
  }
  return forces;
}

// Return array with math function entries
template <unsigned Tdim>
std::array<std::vector<double>, 2> mpm::IOMeshAscii<Tdim>::read_math_functions(
    const std::string& math_file) {
  // Initialise vector with 2 empty vectors
  std::array<std::vector<double>, 2> xfx_values;

  // Read from csv file
  try {
    io::CSVReader<2> in(math_file.c_str());
    double x_value, fx_value;
    while (in.read_row(x_value, fx_value)) {
      xfx_values[0].push_back(x_value);
      xfx_values[1].push_back(fx_value);
    }
  } catch (std::exception& exception) {
    console_->error("Read math functions: {}", exception.what());
  }

  return xfx_values;
}