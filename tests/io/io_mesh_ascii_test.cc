#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "io_mesh_ascii.h"

//! \brief Check IOMeshAscii for 2D
TEST_CASE("IOMeshAscii is checked for 2D", "[IOMesh][IOMeshAscii][2D]") {

  // Dimension
  const unsigned dim = 2;
  // Tolerance
  const double Tolerance = 1.E-7;

  SECTION("Check mesh file ") {
    // Vector of nodal coordinates
    std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

    // Nodal coordinates
    Eigen::Matrix<double, dim, 1> node;

    // Cell 0
    // Node 0
    node << 0., 0.;
    coordinates.emplace_back(node);
    // Node 1
    node << 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 2
    node << 0.5, 0.5;
    coordinates.emplace_back(node);
    // Node 3
    node << 0., 0.5;
    coordinates.emplace_back(node);

    // Cell 1
    // Node 4
    node << 1.0, 0.;
    coordinates.emplace_back(node);
    // Node 5
    node << 1.0, 0.5;
    coordinates.emplace_back(node);

    // Cell with node ids
    std::vector<std::vector<unsigned>> cells{// cell #0
                                             {0, 1, 2, 3},
                                             // cell #1
                                             {1, 4, 5, 2}};

    // Dump mesh file as an input file to be read
    std::ofstream file;
    file.open("mesh-2d.txt");
    file << "! elementShape hexahedron\n";
    file << "! elementNumPoints 8\n";
    file << coordinates.size() << "\t" << cells.size() << "\n";

    // Write nodal coordinates
    for (const auto& coord : coordinates) {
      for (unsigned i = 0; i < coord.size(); ++i) file << coord[i] << "\t";
      file << "\n";
    }

    // Write cell node ids
    for (const auto& cell : cells) {
      for (auto nid : cell) file << nid << "\t";
      file << "\n";
    }

    file.close();

    // Check read mesh nodes
    SECTION("Check read mesh nodes") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read nodes from a non-existant file
      auto check_coords = io_mesh->read_mesh_nodes("mesh-missing.txt");
      // Check number of nodal coordinates
      REQUIRE(check_coords.size() == 0);

      check_coords = io_mesh->read_mesh_nodes("mesh-2d.txt");
      // Check number of nodal coordinates
      REQUIRE(check_coords.size() == coordinates.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < coordinates.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(check_coords[i][j] ==
                  Approx(coordinates[i][j]).epsilon(Tolerance));
        }
      }
    }

    // Check read mesh cells
    SECTION("Check read mesh cell ids") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read cells from a non-existant file
      auto check_node_ids = io_mesh->read_mesh_cells("mesh-missing.txt");
      // Check number of cells
      REQUIRE(check_node_ids.size() == 0);

      // Check node ids in cell
      check_node_ids = io_mesh->read_mesh_cells("mesh-2d.txt");
      // Check number of cells
      REQUIRE(check_node_ids.size() == cells.size());
      // Check node ids of cells
      for (unsigned i = 0; i < cells.size(); ++i) {
        for (unsigned j = 0; j < cells[i].size(); ++j) {
          REQUIRE(check_node_ids[i][j] == cells[i][j]);
        }
      }
    }
  }

  SECTION("Check particles file") {
    // Vector of particle coordinates
    std::vector<Eigen::Matrix<double, dim, 1>> coordinates;
    coordinates.clear();

    // Particle coordinates
    Eigen::Matrix<double, dim, 1> particle;

    // Cell 0
    // Particle 0
    particle << 0.125, 0.125;
    coordinates.emplace_back(particle);
    // Particle 1
    particle << 0.375, 0.125;
    coordinates.emplace_back(particle);
    // Particle 2
    particle << 0.375, 0.375;
    coordinates.emplace_back(particle);
    // Particle 3
    particle << 0.125, 0.375;
    coordinates.emplace_back(particle);

    // Cell 1
    // Particle 4
    particle << 0.625, 0.125;
    coordinates.emplace_back(particle);
    // Particle 5
    particle << 0.875, 0.125;
    coordinates.emplace_back(particle);
    // Particle 6
    particle << 0.875, 0.375;
    coordinates.emplace_back(particle);
    // Particle 7
    particle << 0.625, 0.375;
    coordinates.emplace_back(particle);

    // Dump particles coordinates as an input file to be read
    std::ofstream file;
    file.open("particles-2d.txt");
    file << coordinates.size() << "\n";
    // Write particle coordinates
    for (const auto& coord : coordinates) {
      for (unsigned i = 0; i < coord.size(); ++i) {
        file << coord[i] << "\t";
      }
      file << "\n";
    }

    file.close();

    // Check read particle coordinates
    SECTION("Check particle coordinates") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particle from a non-existant file
      auto particles = io_mesh->read_particles("particles-missing.txt");
      // Check number of particles
      REQUIRE(particles.size() == 0);

      // Check particle coordinates
      particles = io_mesh->read_particles("particles-2d.txt");
      // Check number of particles
      REQUIRE(particles.size() == coordinates.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < coordinates.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(particles[i][j] ==
                  Approx(coordinates[i][j]).epsilon(Tolerance));
        }
      }
    }
  }

  SECTION("Check levelset file") {
    // Vector of levelset inputs
    std::vector<std::tuple<mpm::Index, double, double, double, double>>
        levelset_input_file;

    // Inputs
    levelset_input_file.emplace_back(
        std::make_tuple(0, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(1, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(2, 1.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(3, 1.00, 0.1, 1000., 1.0E+06));

    // Dump levelset inputs as a file to be read
    std::ofstream file;
    file.open("levelset-nodal-values-2d.txt");
    // Write particle coordinates
    for (const auto& levelset_inputs : levelset_input_file) {
      file << std::get<0>(levelset_inputs) << "\t";
      file << std::get<1>(levelset_inputs) << "\t";
      file << std::get<2>(levelset_inputs) << "\t";
      file << std::get<3>(levelset_inputs) << "\t";
      file << std::get<4>(levelset_inputs) << "\t";

      file << "\n";
    }

    file.close();
  }

  // Check read levelset inputs
  SECTION("Check levelset inputs") {
    // Create a read_mesh object
    auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

    // Try to read inputs from a non-existant file
    auto levelset_inputs =
        read_mesh->read_levelset_input("levelset-nodal-values-missing.txt");
    // Check number of inputs
    REQUIRE(levelset_inputs.size() == 0);

    // Check inputs
    levelset_inputs =
        read_mesh->read_levelset_input("levelset-nodal-values-2d.txt");
    // Check number of particles
    REQUIRE(levelset_inputs.size() == levelset_inputs.size());

    // Check coordinates of nodes
    for (unsigned i = 0; i < levelset_inputs.size(); ++i) {
      REQUIRE(std::get<0>(levelset_inputs.at(i)) ==
              Approx(std::get<0>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<1>(levelset_inputs.at(i)) ==
              Approx(std::get<1>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<2>(levelset_inputs.at(i)) ==
              Approx(std::get<2>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<3>(levelset_inputs.at(i)) ==
              Approx(std::get<3>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<4>(levelset_inputs.at(i)) ==
              Approx(std::get<4>(levelset_inputs.at(i))).epsilon(Tolerance));
    }
  }

  SECTION("Check displacement constraints file") {
    // Vector of particle coordinates
    std::vector<std::tuple<mpm::Index, unsigned, double>>
        displacement_constraints;

    // Constraint
    displacement_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    displacement_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    displacement_constraints.emplace_back(std::make_tuple(2, 2, -12.5));
    displacement_constraints.emplace_back(std::make_tuple(3, 0, 0.0));

    // Dump constratints as an input file to be read
    std::ofstream file;
    file.open("displacement-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& displacement_constraint : displacement_constraints) {
      file << std::get<0>(displacement_constraint) << "\t";
      file << std::get<1>(displacement_constraint) << "\t";
      file << std::get<2>(displacement_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read displacement constraints
    SECTION("Check displacement constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constrtaints from a non-existant file
      auto constraints = read_mesh->read_displacement_constraints(
          "displacement-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints = read_mesh->read_displacement_constraints(
          "displacement-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == displacement_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < displacement_constraints.size(); ++i) {
        REQUIRE(std::get<0>(constraints.at(i)) ==
                Approx(std::get<0>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<1>(constraints.at(i)) ==
                Approx(std::get<1>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<2>(constraints.at(i)) ==
                Approx(std::get<2>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
      }
    }
  }

  SECTION("Check velocity constraints file") {
    // Vector of velocity constraints
    std::vector<std::tuple<mpm::Index, unsigned, double>> velocity_constraints;

    // Constraint
    velocity_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    velocity_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    velocity_constraints.emplace_back(std::make_tuple(2, 0, -12.5));
    velocity_constraints.emplace_back(std::make_tuple(3, 1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("velocity-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& velocity_constraint : velocity_constraints) {
      file << std::get<0>(velocity_constraint) << "\t";
      file << std::get<1>(velocity_constraint) << "\t";
      file << std::get<2>(velocity_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read velocity constraints
    SECTION("Check velocity constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_velocity_constraints(
          "velocity-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_velocity_constraints("velocity-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == velocity_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < velocity_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(velocity_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(velocity_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(velocity_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check acceleration constraints file") {
    // Vector of acceleration constraints
    std::vector<std::tuple<mpm::Index, unsigned, double>>
        acceleration_constraints;

    // Constraint
    acceleration_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    acceleration_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    acceleration_constraints.emplace_back(std::make_tuple(2, 0, -12.5));
    acceleration_constraints.emplace_back(std::make_tuple(3, 1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("acceleration-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& acceleration_constraint : acceleration_constraints) {
      file << std::get<0>(acceleration_constraint) << "\t";
      file << std::get<1>(acceleration_constraint) << "\t";
      file << std::get<2>(acceleration_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read acceleration constraints
    SECTION("Check acceleration constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_acceleration_constraints(
          "acceleration-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints = read_mesh->read_acceleration_constraints(
          "acceleration-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == acceleration_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < acceleration_constraints.size(); ++i) {
        REQUIRE(std::get<0>(constraints.at(i)) ==
                Approx(std::get<0>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<1>(constraints.at(i)) ==
                Approx(std::get<1>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<2>(constraints.at(i)) ==
                Approx(std::get<2>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
      }
    }
  }

  SECTION("Check friction constraints file") {
    // Vector of friction constraints
    std::vector<std::tuple<mpm::Index, unsigned, int, double>>
        friction_constraints;

    // Constraint
    friction_constraints.emplace_back(std::make_tuple(0, 0, 1, 0.5));
    friction_constraints.emplace_back(std::make_tuple(1, 1, -1, 0.5));
    friction_constraints.emplace_back(std::make_tuple(2, 0, 1, 0.25));
    friction_constraints.emplace_back(std::make_tuple(3, 1, -1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("friction-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& friction_constraint : friction_constraints) {
      file << std::get<0>(friction_constraint) << "\t";
      file << std::get<1>(friction_constraint) << "\t";
      file << std::get<2>(friction_constraint) << "\t";
      file << std::get<3>(friction_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read friction constraints
    SECTION("Check friction constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_friction_constraints(
          "friction-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_friction_constraints("friction-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == friction_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < friction_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<3>(constraints.at(i)) ==
            Approx(std::get<3>(friction_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check adhesion constraints file") {
    // Vector of adhesion constraints
    std::vector<std::tuple<mpm::Index, unsigned, int, double, double, int>>
        adhesion_constraints;

    // Constraint
    adhesion_constraints.emplace_back(std::make_tuple(0, 0, -1, 100, 0.25, 1));
    adhesion_constraints.emplace_back(std::make_tuple(1, 0, -1, 100, 0.25, 2));
    adhesion_constraints.emplace_back(std::make_tuple(2, 1, -1, 100, 0.25, 2));
    adhesion_constraints.emplace_back(std::make_tuple(3, 1, -1, 100, 0.25, 1));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("adhesion-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& adhesion_constraint : adhesion_constraints) {
      file << std::get<0>(adhesion_constraint) << "\t";
      file << std::get<1>(adhesion_constraint) << "\t";
      file << std::get<2>(adhesion_constraint) << "\t";
      file << std::get<3>(adhesion_constraint) << "\t";
      file << std::get<4>(adhesion_constraint) << "\t";
      file << std::get<5>(adhesion_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read adhesion constraints
    SECTION("Check adhesion constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_adhesion_constraints(
          "adhesion-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_adhesion_constraints("adhesion-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == adhesion_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < adhesion_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<3>(constraints.at(i)) ==
            Approx(std::get<3>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<4>(constraints.at(i)) ==
            Approx(std::get<4>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<5>(constraints.at(i)) ==
            Approx(std::get<5>(adhesion_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  // Check nodal pressure constraints file
  SECTION("Check pressure constraints file") {
    // Vector of pressure constraints
    std::vector<std::tuple<mpm::Index, double>> pressure_constraints;

    // Pressure constraint
    pressure_constraints.emplace_back(std::make_tuple(0, 300.5));
    pressure_constraints.emplace_back(std::make_tuple(1, 500.5));
    pressure_constraints.emplace_back(std::make_tuple(2, 250.5));
    pressure_constraints.emplace_back(std::make_tuple(3, 0.0));

    // Dump pressure constraints as an input file to be read
    std::ofstream file;
    file.open("pressure-constraints-2d.txt");
    // Write particle coordinates
    for (const auto& pressure_constraint : pressure_constraints) {
      file << std::get<0>(pressure_constraint) << "\t";
      file << std::get<1>(pressure_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read pressure constraints
    SECTION("Check pressure constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read pressure constraints from a non-existant file
      auto constraints = read_mesh->read_pressure_constraints(
          "pressure-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_pressure_constraints("pressure-constraints-2d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == pressure_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < pressure_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(pressure_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(pressure_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check forces file") {
    // Vector of particle forces
    std::vector<std::tuple<mpm::Index, unsigned, double>> nodal_forces;

    // Constraint
    nodal_forces.emplace_back(std::make_tuple(0, 0, 10.5));
    nodal_forces.emplace_back(std::make_tuple(1, 1, -10.5));
    nodal_forces.emplace_back(std::make_tuple(2, 0, -12.5));
    nodal_forces.emplace_back(std::make_tuple(3, 1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("forces-2d.txt");
    // Write particle coordinates
    for (const auto& force : nodal_forces) {
      file << std::get<0>(force) << "\t";
      file << std::get<1>(force) << "\t";
      file << std::get<2>(force) << "\t";

      file << "\n";
    }

    file.close();

    // Check read forces
    SECTION("Check forces") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constrtaints from a non-existant file
      auto forces = read_mesh->read_forces("forces-missing.txt");
      // Check number of forces
      REQUIRE(forces.size() == 0);

      // Check forces
      forces = read_mesh->read_forces("forces-2d.txt");
      // Check number of nodes
      REQUIRE(forces.size() == nodal_forces.size());

      // Check forces
      for (unsigned i = 0; i < nodal_forces.size(); ++i) {
        REQUIRE(std::get<0>(forces.at(i)) ==
                Approx(std::get<0>(nodal_forces.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<1>(forces.at(i)) ==
                Approx(std::get<1>(nodal_forces.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<2>(forces.at(i)) ==
                Approx(std::get<2>(nodal_forces.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check nodal euler angle file") {
    // Map of euler angles
    std::map<mpm::Index, Eigen::Matrix<double, dim, 1>> euler_angles;
    euler_angles.emplace(
        std::make_pair(0, (Eigen::Matrix<double, dim, 1>(10.5, 20.5))));
    euler_angles.emplace(
        std::make_pair(1, (Eigen::Matrix<double, dim, 1>(30.5, -40.5))));
    euler_angles.emplace(
        std::make_pair(2, (Eigen::Matrix<double, dim, 1>(-50.5, -60.5))));
    euler_angles.emplace(
        std::make_pair(3, (Eigen::Matrix<double, dim, 1>(-70.5, 80.5))));

    // Dump euler angles as an input file to be read
    std::ofstream file;
    file.open("nodal-euler-angles-2d.txt");
    // Write particle coordinates
    for (const auto& angles : euler_angles) {
      file << angles.first << "\t";
      for (unsigned i = 0; i < dim; ++i) file << (angles.second)(i) << "\t";
      file << "\n";
    }

    file.close();

    // Check read euler angles
    SECTION("Check euler angles") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read euler angles from a non-existant file
      auto read_euler_angles =
          io_mesh->read_euler_angles("nodal-euler-angles-missing.txt");
      // Check number of euler angles
      REQUIRE(read_euler_angles.size() == 0);

      // Check euler angles
      read_euler_angles =
          io_mesh->read_euler_angles("nodal-euler-angles-2d.txt");

      // Check number of elements
      REQUIRE(read_euler_angles.size() == euler_angles.size());

      // Check euler angles
      for (unsigned i = 0; i < euler_angles.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(read_euler_angles.at(i)(j) ==
                  Approx(euler_angles.at(i)(j)).epsilon(Tolerance));
        }
      }
    }
  }

  SECTION("Check particles volume file") {
    // Map of particle volumes
    std::vector<std::tuple<mpm::Index, double>> particles_volumes;
    particles_volumes.emplace_back(std::make_tuple(0, 1.5));
    particles_volumes.emplace_back(std::make_tuple(1, 2.5));
    particles_volumes.emplace_back(std::make_tuple(2, 3.5));
    particles_volumes.emplace_back(std::make_tuple(3, 0.0));

    // Dump particle volumes as an input file to be read
    std::ofstream file;
    file.open("particles-volumes-2d.txt");
    // Write particle volumes
    for (const auto& particles_volume : particles_volumes) {
      file << std::get<0>(particles_volume) << "\t";
      file << std::get<1>(particles_volume) << "\t";

      file << "\n";
    }

    file.close();

    // Check read particles volumes
    SECTION("Check particles volumes") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles volumes from a non-existant file
      auto read_volumes =
          io_mesh->read_particles_volumes("particles-volumes-missing.txt");
      // Check number of particle volumes
      REQUIRE(read_volumes.size() == 0);

      // Check particles volumes
      read_volumes =
          io_mesh->read_particles_volumes("particles-volumes-2d.txt");

      // Check number of elements
      REQUIRE(read_volumes.size() == particles_volumes.size());

      // Check particles volumes
      for (unsigned i = 0; i < particles_volumes.size(); ++i) {
        REQUIRE(
            std::get<0>(read_volumes.at(i)) ==
            Approx(std::get<0>(particles_volumes.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(read_volumes.at(i)) ==
            Approx(std::get<1>(particles_volumes.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check particles cells file") {
    // Map of particle volumes
    std::vector<std::tuple<mpm::Index, mpm::Index>> particles_cells;
    particles_cells.emplace_back(std::make_tuple(0, 0));
    particles_cells.emplace_back(std::make_tuple(1, 0));
    particles_cells.emplace_back(std::make_tuple(2, 1));
    particles_cells.emplace_back(std::make_tuple(3, 1));

    // Dump particle cells as an input file to be read
    std::ofstream file;
    file.open("particles-cells-2d.txt");
    // Write particle coordinates
    for (const auto& particles_cell : particles_cells) {
      file << std::get<0>(particles_cell) << "\t";
      file << std::get<1>(particles_cell) << "\t";

      file << "\n";
    }

    file.close();

    // Check read particles cells
    SECTION("Check particles cells") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles cells from a non-existant file
      auto read_cells =
          io_mesh->read_particles_cells("particles-cells-missing.txt");
      // Check number of particle cells
      REQUIRE(read_cells.size() == 0);

      // Check particles cells
      read_cells = io_mesh->read_particles_cells("particles-cells-2d.txt");

      // Check number of elements
      REQUIRE(read_cells.size() == particles_cells.size());

      // Check particles cells
      for (unsigned i = 0; i < particles_cells.size(); ++i) {
        REQUIRE(std::get<0>(read_cells.at(i)) ==
                Approx(std::get<0>(particles_cells.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<1>(read_cells.at(i)) ==
                Approx(std::get<1>(particles_cells.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check tractions file") {
    // Vector of particle tractions
    std::vector<std::tuple<mpm::Index, unsigned, double>> particles_tractions;

    // Constraint
    particles_tractions.emplace_back(std::make_tuple(0, 0, 10.5));
    particles_tractions.emplace_back(std::make_tuple(1, 1, -10.5));
    particles_tractions.emplace_back(std::make_tuple(2, 0, -12.5));
    particles_tractions.emplace_back(std::make_tuple(3, 1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("tractions-2d.txt");
    // Write particle coordinates
    for (const auto& traction : particles_tractions) {
      file << std::get<0>(traction) << "\t";
      file << std::get<1>(traction) << "\t";
      file << std::get<2>(traction) << "\t";

      file << "\n";
    }

    file.close();
  }

  SECTION("Check stresses file") {
    // Vector of particle stresses
    std::vector<Eigen::Matrix<double, 6, 1>> particles_stresses;

    // Stresses
    particles_stresses.emplace_back(
        Eigen::Matrix<double, 6, 1>::Constant(10.5));
    particles_stresses.emplace_back(
        Eigen::Matrix<double, 6, 1>::Constant(-12.5));
    particles_stresses.emplace_back(Eigen::Matrix<double, 6, 1>::Constant(0.4));

    // Dump initial stresses as an input file to be read
    std::ofstream file;
    file.open("particle-stresses-2d.txt");
    file << particles_stresses.size() << "\n";
    // Write particle coordinates
    for (const auto& stress : particles_stresses) {
      for (unsigned i = 0; i < stress.size(); ++i) file << stress[i] << "\t";
      file << "\n";
    }

    file.close();

    // Check read stresses
    SECTION("Check stresses") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read stresses from a non-existant file
      auto stresses = io_mesh->read_particles_stresses("stresses-missing.txt");
      // Check number of stresses
      REQUIRE(stresses.size() == 0);

      // Check stresses
      stresses = io_mesh->read_particles_stresses("particle-stresses-2d.txt");
      // Check number of particles
      REQUIRE(stresses.size() == particles_stresses.size());

      // Check stresses
      for (unsigned i = 0; i < particles_stresses.size(); ++i) {
        for (unsigned j = 0; j < particles_stresses.at(0).size(); ++j)
          REQUIRE(stresses.at(i)[j] ==
                  Approx(particles_stresses.at(i)[j]).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check particles scalar properties file") {
    // Particle scalar properties
    std::map<mpm::Index, double> particles_scalars;
    particles_scalars.emplace(std::make_pair(0, 10.5));
    particles_scalars.emplace(std::make_pair(1, -40.5));
    particles_scalars.emplace(std::make_pair(2, -60.5));
    particles_scalars.emplace(std::make_pair(3, 80.5));

    // Dump particle scalar properties as an input file to be read
    std::ofstream file;
    file.open("particles-scalars-2d.txt");
    // Write particle scalar properties
    for (const auto& scalars : particles_scalars) {
      file << scalars.first << "\t";
      file << scalars.second << "\n";
    }

    file.close();

    // Check read particles scalar properties file
    SECTION("Check read particles scalar properties file") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles scalar properties from a non-existant file
      auto read_particles_scalars =
          io_mesh->read_scalar_properties("particles-scalar-missing.txt");
      // Check number of particles scalar properties
      REQUIRE(read_particles_scalars.size() == 0);

      // Check particles scalar properties
      read_particles_scalars =
          io_mesh->read_scalar_properties("particles-scalars-2d.txt");

      // Check number of particles
      REQUIRE(read_particles_scalars.size() == particles_scalars.size());

      // Check particles scalar properties
      for (unsigned i = 0; i < particles_scalars.size(); ++i) {
        REQUIRE(std::get<1>(read_particles_scalars.at(i)) ==
                Approx(particles_scalars.at(i)).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check math function file") {
    // Vector of math function entries
    std::array<std::vector<double>, 2> entries;

    // Populate the math function entries
    for (int i = 0; i < 3; ++i) {
      entries[0].push_back(0.7 * i);
      entries[1].push_back(2.2 * i);
    }

    // Dump the math entries as an input file to be read
    std::ofstream file;
    file.open("math-function-2d.csv");
    // Write math entries for x and fx
    for (int i = 0; i < 3; ++i)
      file << entries[0][i] << "," << entries[1][i] << "\n";

    file.close();

    // Check read math funciton file
    SECTION("Check math function entries") {
      // Create an io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read math funciton entries from a non-existant file
      auto math_function_values =
          io_mesh->read_math_functions("math-function-missing.csv");
      // Check number of math function entries
      REQUIRE(math_function_values[0].size() == 0);
      REQUIRE(math_function_values[1].size() == 0);

      // Check math function reading
      math_function_values =
          io_mesh->read_math_functions("math-function-2d.csv");
      // Check number of entries from math function file
      REQUIRE(math_function_values[0].size() == entries[0].size());
      REQUIRE(math_function_values[1].size() == entries[1].size());
      REQUIRE(math_function_values[0].size() == math_function_values[1].size());

      // Check entry values
      for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < entries[0].size(); ++j)
          REQUIRE(math_function_values[i][j] ==
                  Approx(entries[i][j]).epsilon(Tolerance));
      }
    }
  }
}

//! \brief Check IOMeshAscii for 3D
TEST_CASE("IOMeshAscii is checked for 3D", "[IOMesh][IOMeshAscii][3D]") {

  // Dimension
  const unsigned dim = 3;
  // Tolerance
  const double Tolerance = 1.E-7;

  SECTION("Check mesh file ") {
    // Vector of nodal coordinates
    std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

    // Nodal coordinates
    Eigen::Matrix<double, dim, 1> node;

    // Cell 0
    // Node 0
    node << 0., 0., 0.;
    coordinates.emplace_back(node);
    // Node 1
    node << 0.5, 0., 0.;
    coordinates.emplace_back(node);
    // Node 2
    node << 0.5, 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 3
    node << 0., 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 4
    node << 0., 0., 0.5;
    coordinates.emplace_back(node);
    // Node 5
    node << 0.5, 0., 0.5;
    coordinates.emplace_back(node);
    // Node 6
    node << 0.5, 0.5, 0.5;
    coordinates.emplace_back(node);
    // Node 7
    node << 0., 0.5, 0.5;
    coordinates.emplace_back(node);

    // Cell 1
    // Node 8
    node << 1.0, 0., 0.;
    coordinates.emplace_back(node);
    // Node 9
    node << 1.0, 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 10
    node << 1.0, 0., 0.5;
    coordinates.emplace_back(node);
    // Node 11
    node << 1.0, 0.5, 0.5;
    coordinates.emplace_back(node);

    // Cell with node ids
    std::vector<std::vector<unsigned>> cells{// cell #0
                                             {0, 1, 2, 3, 4, 5, 6, 7},
                                             // cell #1
                                             {1, 8, 9, 2, 5, 10, 11, 6}};

    // Dump mesh file as an input file to be read
    std::ofstream file;
    file.open("mesh-3d.txt");
    file << "! elementShape hexahedron\n";
    file << "! elementNumPoints 8\n";
    file << coordinates.size() << "\t" << cells.size() << "\n";

    // Write nodal coordinates
    for (const auto& coord : coordinates) {
      for (unsigned i = 0; i < coord.size(); ++i) file << coord[i] << "\t";
      file << "\n";
    }

    // Write cell node ids
    for (const auto& cell : cells) {
      for (auto nid : cell) file << nid << "\t";
      file << "\n";
    }

    file.close();

    // Check read mesh nodes
    SECTION("Check read mesh nodes") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Check nodal coordinates
      auto check_coords = io_mesh->read_mesh_nodes("mesh-missing.txt");
      // Check number of nodal coordinates
      REQUIRE(check_coords.size() == 0);

      // Check nodal coordinates
      check_coords = io_mesh->read_mesh_nodes("mesh-3d.txt");
      // Check number of nodal coordinates
      REQUIRE(check_coords.size() == coordinates.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < coordinates.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(check_coords[i][j] ==
                  Approx(coordinates[i][j]).epsilon(Tolerance));
        }
      }
    }

    // Check read mesh cells
    SECTION("Check read mesh cell ids") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read cells from a non-existant file
      auto check_node_ids = io_mesh->read_mesh_cells("mesh-missing.txt");
      // Check number of cells
      REQUIRE(check_node_ids.size() == 0);

      // Check node ids in cell
      check_node_ids = io_mesh->read_mesh_cells("mesh-3d.txt");
      // Check number of cells
      REQUIRE(check_node_ids.size() == cells.size());
      // Check node ids of cells
      for (unsigned i = 0; i < cells.size(); ++i) {
        for (unsigned j = 0; j < cells[i].size(); ++j) {
          REQUIRE(check_node_ids[i][j] == cells[i][j]);
        }
      }
    }
  }

  SECTION("Check particles file") {
    // Vector of particle coordinates
    std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

    // Particle coordinates
    Eigen::Matrix<double, dim, 1> particle;

    // Cell 0
    // Particle 0
    particle << 0.125, 0.125, 0.125;
    coordinates.emplace_back(particle);
    // Particle 1
    particle << 0.375, 0.125, 0.125;
    coordinates.emplace_back(particle);
    // Particle 2
    particle << 0.375, 0.375, 0.125;
    coordinates.emplace_back(particle);
    // Particle 3
    particle << 0.125, 0.375, 0.125;
    coordinates.emplace_back(particle);
    // Particle 4
    particle << 0.125, 0.125, 0.375;
    coordinates.emplace_back(particle);
    // Particle 5
    particle << 0.375, 0.125, 0.375;
    coordinates.emplace_back(particle);
    // Particle 6
    particle << 0.375, 0.375, 0.375;
    coordinates.emplace_back(particle);
    // Particle 7
    particle << 0.125, 0.375, 0.375;
    coordinates.emplace_back(particle);

    // Cell 1
    // Particle 8
    particle << 0.625, 0.125, 0.125;
    coordinates.emplace_back(particle);
    // Particle 9
    particle << 0.875, 0.125, 0.125;
    coordinates.emplace_back(particle);
    // Particle 10
    particle << 0.875, 0.375, 0.125;
    coordinates.emplace_back(particle);
    // Particle 11
    particle << 0.625, 0.375, 0.125;
    coordinates.emplace_back(particle);
    // Particle 12
    particle << 0.675, 0.125, 0.375;
    coordinates.emplace_back(particle);
    // Particle 13
    particle << 0.875, 0.125, 0.375;
    coordinates.emplace_back(particle);
    // Particle 14
    particle << 0.875, 0.375, 0.375;
    coordinates.emplace_back(particle);
    // Particle 15
    particle << 0.675, 0.375, 0.375;
    coordinates.emplace_back(particle);

    // Dump particles coordinates as an input file to be read
    std::ofstream file;
    file.open("particles-3d.txt");
    file << coordinates.size() << "\n";
    // Write particle coordinates
    for (const auto& coord : coordinates) {
      for (unsigned i = 0; i < coord.size(); ++i) {
        file << coord[i] << "\t";
      }
      file << "\n";
    }

    file.close();

    // Check read particles
    SECTION("Check particle coordinates") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particle from a non-existant file
      auto particles = io_mesh->read_particles("particles-missing.txt");
      // Check number of particles
      REQUIRE(particles.size() == 0);

      // Check particle coordinates
      particles = io_mesh->read_particles("particles-3d.txt");
      // Check number of particles
      REQUIRE(particles.size() == coordinates.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < coordinates.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(particles[i][j] ==
                  Approx(coordinates[i][j]).epsilon(Tolerance));
        }
      }
    }
  }

  SECTION("Check levelset file") {
    // Vector of levelset inputs
    std::vector<std::tuple<mpm::Index, double, double, double, double>>
        levelset_input_file;

    // Inputs
    levelset_input_file.emplace_back(
        std::make_tuple(0, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(1, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(2, 1.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(3, 1.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(4, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(5, 0.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(6, 1.00, 0.1, 1000., 1.0E+06));
    levelset_input_file.emplace_back(
        std::make_tuple(7, 1.00, 0.1, 1000., 1.0E+06));

    // Dump levelset inputs as a file to be read
    std::ofstream file;
    file.open("levelset-nodal-values-3d.txt");
    // Write particle coordinates
    for (const auto& levelset_inputs : levelset_input_file) {
      file << std::get<0>(levelset_inputs) << "\t";
      file << std::get<1>(levelset_inputs) << "\t";
      file << std::get<2>(levelset_inputs) << "\t";
      file << std::get<3>(levelset_inputs) << "\t";
      file << std::get<4>(levelset_inputs) << "\t";

      file << "\n";
    }

    file.close();
  }

  // Check read levelset inputs
  SECTION("Check levelset inputs") {
    // Create a read_mesh object
    auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

    // Try to read inputs from a non-existant file
    auto levelset_inputs =
        read_mesh->read_levelset_input("levelset-nodal-values-missing.txt");
    // Check number of inputs
    REQUIRE(levelset_inputs.size() == 0);

    // Check inputs
    levelset_inputs =
        read_mesh->read_levelset_input("levelset-nodal-values-3d.txt");
    // Check number of particles
    REQUIRE(levelset_inputs.size() == levelset_inputs.size());

    // Check coordinates of nodes
    for (unsigned i = 0; i < levelset_inputs.size(); ++i) {
      REQUIRE(std::get<0>(levelset_inputs.at(i)) ==
              Approx(std::get<0>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<1>(levelset_inputs.at(i)) ==
              Approx(std::get<1>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<2>(levelset_inputs.at(i)) ==
              Approx(std::get<2>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<3>(levelset_inputs.at(i)) ==
              Approx(std::get<3>(levelset_inputs.at(i))).epsilon(Tolerance));
      REQUIRE(std::get<4>(levelset_inputs.at(i)) ==
              Approx(std::get<4>(levelset_inputs.at(i))).epsilon(Tolerance));
    }
  }

  SECTION("Check displacement constraints file") {
    // Vector of particle coordinates
    std::vector<std::tuple<mpm::Index, unsigned, double>>
        displacement_constraints;

    // Constraint
    displacement_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    displacement_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    displacement_constraints.emplace_back(std::make_tuple(2, 2, -12.5));
    displacement_constraints.emplace_back(std::make_tuple(3, 0, 0.0));

    // Dump constratints as an input file to be read
    std::ofstream file;
    file.open("displacement-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& displacement_constraint : displacement_constraints) {
      file << std::get<0>(displacement_constraint) << "\t";
      file << std::get<1>(displacement_constraint) << "\t";
      file << std::get<2>(displacement_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read displacement constraints
    SECTION("Check displacement constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constrtaints from a non-existant file
      auto constraints = read_mesh->read_displacement_constraints(
          "displacement-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints = read_mesh->read_displacement_constraints(
          "displacement-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == displacement_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < displacement_constraints.size(); ++i) {
        REQUIRE(std::get<0>(constraints.at(i)) ==
                Approx(std::get<0>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<1>(constraints.at(i)) ==
                Approx(std::get<1>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<2>(constraints.at(i)) ==
                Approx(std::get<2>(displacement_constraints.at(i)))
                    .epsilon(Tolerance));
      }
    }
  }

  SECTION("Check velocity constraints file") {
    // Vector of particle coordinates
    std::vector<std::tuple<mpm::Index, unsigned, double>> velocity_constraints;

    // Constraint
    velocity_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    velocity_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    velocity_constraints.emplace_back(std::make_tuple(2, 2, -12.5));
    velocity_constraints.emplace_back(std::make_tuple(3, 0, 0.0));

    // Dump constratints as an input file to be read
    std::ofstream file;
    file.open("velocity-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& velocity_constraint : velocity_constraints) {
      file << std::get<0>(velocity_constraint) << "\t";
      file << std::get<1>(velocity_constraint) << "\t";
      file << std::get<2>(velocity_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read velocity constraints
    SECTION("Check velocity constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constrtaints from a non-existant file
      auto constraints = read_mesh->read_velocity_constraints(
          "velocity-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_velocity_constraints("velocity-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == velocity_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < velocity_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(velocity_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(velocity_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(velocity_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check acceleration constraints file") {
    // Vector of particle coordinates
    std::vector<std::tuple<mpm::Index, unsigned, double>>
        acceleration_constraints;

    // Constraint
    acceleration_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
    acceleration_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
    acceleration_constraints.emplace_back(std::make_tuple(2, 2, -12.5));
    acceleration_constraints.emplace_back(std::make_tuple(3, 0, 0.0));

    // Dump constratints as an input file to be read
    std::ofstream file;
    file.open("acceleration-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& acceleration_constraint : acceleration_constraints) {
      file << std::get<0>(acceleration_constraint) << "\t";
      file << std::get<1>(acceleration_constraint) << "\t";
      file << std::get<2>(acceleration_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read acceleration constraints
    SECTION("Check acceleration constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_acceleration_constraints(
          "acceleration-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints = read_mesh->read_acceleration_constraints(
          "acceleration-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == acceleration_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < acceleration_constraints.size(); ++i) {
        REQUIRE(std::get<0>(constraints.at(i)) ==
                Approx(std::get<0>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<1>(constraints.at(i)) ==
                Approx(std::get<1>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
        REQUIRE(std::get<2>(constraints.at(i)) ==
                Approx(std::get<2>(acceleration_constraints.at(i)))
                    .epsilon(Tolerance));
      }
    }
  }

  SECTION("Check friction constraints file") {
    // Vector of friction constraints
    std::vector<std::tuple<mpm::Index, unsigned, int, double>>
        friction_constraints;

    // Constraint
    friction_constraints.emplace_back(std::make_tuple(0, 0, 1, 0.5));
    friction_constraints.emplace_back(std::make_tuple(1, 1, -1, 0.5));
    friction_constraints.emplace_back(std::make_tuple(2, 0, 1, 0.25));
    friction_constraints.emplace_back(std::make_tuple(3, 2, -1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("friction-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& friction_constraint : friction_constraints) {
      file << std::get<0>(friction_constraint) << "\t";
      file << std::get<1>(friction_constraint) << "\t";
      file << std::get<2>(friction_constraint) << "\t";
      file << std::get<3>(friction_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read friction constraints
    SECTION("Check friction constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_friction_constraints(
          "friction-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_friction_constraints("friction-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == friction_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < friction_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(friction_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<3>(constraints.at(i)) ==
            Approx(std::get<3>(friction_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check adhesion constraints file") {
    // Vector of adhesion constraints
    std::vector<std::tuple<mpm::Index, unsigned, int, double, double, int>>
        adhesion_constraints;

    // Constraint
    adhesion_constraints.emplace_back(std::make_tuple(0, 1, -1, 100, 0.25, 1));
    adhesion_constraints.emplace_back(std::make_tuple(1, 1, -1, 100, 0.25, 2));
    adhesion_constraints.emplace_back(std::make_tuple(2, 1, -1, 100, 0.25, 3));
    adhesion_constraints.emplace_back(std::make_tuple(3, 2, -1, 100, 0.25, 3));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("adhesion-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& adhesion_constraint : adhesion_constraints) {
      file << std::get<0>(adhesion_constraint) << "\t";
      file << std::get<1>(adhesion_constraint) << "\t";
      file << std::get<2>(adhesion_constraint) << "\t";
      file << std::get<3>(adhesion_constraint) << "\t";
      file << std::get<4>(adhesion_constraint) << "\t";
      file << std::get<5>(adhesion_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read adhesion constraints
    SECTION("Check adhesion constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constraints from a non-existant file
      auto constraints = read_mesh->read_adhesion_constraints(
          "adhesion-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_adhesion_constraints("adhesion-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == adhesion_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < adhesion_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<2>(constraints.at(i)) ==
            Approx(std::get<2>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<3>(constraints.at(i)) ==
            Approx(std::get<3>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<4>(constraints.at(i)) ==
            Approx(std::get<4>(adhesion_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<5>(constraints.at(i)) ==
            Approx(std::get<5>(adhesion_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  // Check nodal pressure constraints file
  SECTION("Check pressure constraints file") {
    // Vector of pressure constraints
    std::vector<std::tuple<mpm::Index, double>> pressure_constraints;

    // Pressure constraint
    pressure_constraints.emplace_back(std::make_tuple(0, 300.5));
    pressure_constraints.emplace_back(std::make_tuple(1, 500.5));
    pressure_constraints.emplace_back(std::make_tuple(2, 250.5));
    pressure_constraints.emplace_back(std::make_tuple(3, 0.0));

    // Dump pressure constraints as an input file to be read
    std::ofstream file;
    file.open("pressure-constraints-3d.txt");
    // Write particle coordinates
    for (const auto& pressure_constraint : pressure_constraints) {
      file << std::get<0>(pressure_constraint) << "\t";
      file << std::get<1>(pressure_constraint) << "\t";

      file << "\n";
    }

    file.close();

    // Check read pressure constraints
    SECTION("Check pressure constraints") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read pressure constraints from a non-existant file
      auto constraints = read_mesh->read_pressure_constraints(
          "pressure-constraints-missing.txt");
      // Check number of constraints
      REQUIRE(constraints.size() == 0);

      // Check constraints
      constraints =
          read_mesh->read_pressure_constraints("pressure-constraints-3d.txt");
      // Check number of particles
      REQUIRE(constraints.size() == pressure_constraints.size());

      // Check coordinates of nodes
      for (unsigned i = 0; i < pressure_constraints.size(); ++i) {
        REQUIRE(
            std::get<0>(constraints.at(i)) ==
            Approx(std::get<0>(pressure_constraints.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(constraints.at(i)) ==
            Approx(std::get<1>(pressure_constraints.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check forces file") {
    // Vector of particle forces
    std::vector<std::tuple<mpm::Index, unsigned, double>> nodal_forces;

    // Constraint
    nodal_forces.emplace_back(std::make_tuple(0, 0, 10.5));
    nodal_forces.emplace_back(std::make_tuple(1, 1, -10.5));
    nodal_forces.emplace_back(std::make_tuple(2, 0, -12.5));
    nodal_forces.emplace_back(std::make_tuple(3, 2, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("forces-3d.txt");
    // Write particle coordinates
    for (const auto& force : nodal_forces) {
      file << std::get<0>(force) << "\t";
      file << std::get<1>(force) << "\t";
      file << std::get<2>(force) << "\t";

      file << "\n";
    }

    file.close();

    // Check read forces
    SECTION("Check forces") {
      // Create a read_mesh object
      auto read_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read constrtaints from a non-existant file
      auto forces = read_mesh->read_forces("forces-missing.txt");
      // Check number of forces
      REQUIRE(forces.size() == 0);

      // Check forces
      forces = read_mesh->read_forces("forces-3d.txt");
      // Check number of nodes
      REQUIRE(forces.size() == nodal_forces.size());

      // Check forces
      for (unsigned i = 0; i < nodal_forces.size(); ++i) {
        REQUIRE(std::get<0>(forces.at(i)) ==
                Approx(std::get<0>(nodal_forces.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<1>(forces.at(i)) ==
                Approx(std::get<1>(nodal_forces.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<2>(forces.at(i)) ==
                Approx(std::get<2>(nodal_forces.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check nodal euler angle file") {
    // Map of euler angles
    std::map<mpm::Index, Eigen::Matrix<double, dim, 1>> euler_angles;
    euler_angles.emplace(
        std::make_pair(0, (Eigen::Matrix<double, dim, 1>(10.5, 20.5, 30.5))));
    euler_angles.emplace(
        std::make_pair(1, (Eigen::Matrix<double, dim, 1>(40.5, -50.5, -60.5))));
    euler_angles.emplace(std::make_pair(
        2, (Eigen::Matrix<double, dim, 1>(-70.5, -80.5, -90.5))));
    euler_angles.emplace(std::make_pair(
        3, (Eigen::Matrix<double, dim, 1>(-100.5, 110.5, 120.5))));

    // Dump euler angles as an input file to be read
    std::ofstream file;
    file.open("nodal-euler-angles-3d.txt");
    // Write particle coordinates
    for (const auto& angles : euler_angles) {
      file << angles.first << "\t";
      for (unsigned i = 0; i < dim; ++i) file << (angles.second)(i) << "\t";
      file << "\n";
    }

    file.close();

    // Check read euler angles
    SECTION("Check euler angles") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read euler angles from a non-existant file
      auto read_euler_angles =
          io_mesh->read_euler_angles("nodal-euler-angles-missing.txt");
      // Check number of euler angles
      REQUIRE(read_euler_angles.size() == 0);

      // Check euler angles
      read_euler_angles =
          io_mesh->read_euler_angles("nodal-euler-angles-3d.txt");

      // Check number of elements
      REQUIRE(read_euler_angles.size() == euler_angles.size());

      // Check euler angles
      for (unsigned i = 0; i < euler_angles.size(); ++i) {
        for (unsigned j = 0; j < dim; ++j) {
          REQUIRE(read_euler_angles.at(i)(j) ==
                  Approx(euler_angles.at(i)(j)).epsilon(Tolerance));
        }
      }
    }
  }

  SECTION("Check particles volume file") {
    // Map of particle volumes
    std::vector<std::tuple<mpm::Index, double>> particles_volumes;
    particles_volumes.emplace_back(std::make_tuple(0, 1.5));
    particles_volumes.emplace_back(std::make_tuple(1, 2.5));
    particles_volumes.emplace_back(std::make_tuple(2, 3.5));
    particles_volumes.emplace_back(std::make_tuple(3, 0.0));

    // Dump particle volumes as an input file to be read
    std::ofstream file;
    file.open("particles-volumes-3d.txt");
    // Write particle volumes
    for (const auto& particles_volume : particles_volumes) {
      file << std::get<0>(particles_volume) << "\t";
      file << std::get<1>(particles_volume) << "\t";

      file << "\n";
    }

    file.close();

    // Check read particles volumes
    SECTION("Check particles volumes") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles volumes from a non-existant file
      auto read_volumes =
          io_mesh->read_particles_volumes("particles-volumes-missing.txt");
      // Check number of particle volumes
      REQUIRE(read_volumes.size() == 0);

      // Check particles volumes
      read_volumes =
          io_mesh->read_particles_volumes("particles-volumes-3d.txt");

      // Check number of elements
      REQUIRE(read_volumes.size() == particles_volumes.size());

      // Check particles volumes
      for (unsigned i = 0; i < particles_volumes.size(); ++i) {
        REQUIRE(
            std::get<0>(read_volumes.at(i)) ==
            Approx(std::get<0>(particles_volumes.at(i))).epsilon(Tolerance));
        REQUIRE(
            std::get<1>(read_volumes.at(i)) ==
            Approx(std::get<1>(particles_volumes.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check particles cells file") {
    // Map of particle volumes
    std::vector<std::tuple<mpm::Index, mpm::Index>> particles_cells;
    particles_cells.emplace_back(std::make_tuple(0, 0));
    particles_cells.emplace_back(std::make_tuple(1, 0));
    particles_cells.emplace_back(std::make_tuple(2, 1));
    particles_cells.emplace_back(std::make_tuple(3, 1));

    // Dump particle cells as an input file to be read
    std::ofstream file;
    file.open("particles-cells-3d.txt");
    // Write particle coordinates
    for (const auto& particles_cell : particles_cells) {
      file << std::get<0>(particles_cell) << "\t";
      file << std::get<1>(particles_cell) << "\t";

      file << "\n";
    }

    file.close();

    // Check read particles cells
    SECTION("Check particles cells") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles cells from a non-existant file
      auto read_cells =
          io_mesh->read_particles_cells("particles-cells-missing.txt");
      // Check number of particle cells
      REQUIRE(read_cells.size() == 0);

      // Check particles cells
      read_cells = io_mesh->read_particles_cells("particles-cells-3d.txt");

      // Check number of elements
      REQUIRE(read_cells.size() == particles_cells.size());

      // Check particles cells
      for (unsigned i = 0; i < particles_cells.size(); ++i) {
        REQUIRE(std::get<0>(read_cells.at(i)) ==
                Approx(std::get<0>(particles_cells.at(i))).epsilon(Tolerance));
        REQUIRE(std::get<1>(read_cells.at(i)) ==
                Approx(std::get<1>(particles_cells.at(i))).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check tractions file") {
    // Vector of particle tractions
    std::vector<std::tuple<mpm::Index, unsigned, double>> particles_tractions;

    // Constraint
    particles_tractions.emplace_back(std::make_tuple(0, 0, 10.5));
    particles_tractions.emplace_back(std::make_tuple(1, 1, -10.5));
    particles_tractions.emplace_back(std::make_tuple(2, 0, -12.5));
    particles_tractions.emplace_back(std::make_tuple(3, 1, 0.0));

    // Dump constraints as an input file to be read
    std::ofstream file;
    file.open("tractions-3d.txt");
    // Write particle coordinates
    for (const auto& traction : particles_tractions) {
      file << std::get<0>(traction) << "\t";
      file << std::get<1>(traction) << "\t";
      file << std::get<2>(traction) << "\t";

      file << "\n";
    }

    file.close();
  }

  SECTION("Check stresses file") {
    // Vector of particle stresses
    std::vector<Eigen::Matrix<double, 6, 1>> particles_stresses;

    // Stresses
    particles_stresses.emplace_back(
        Eigen::Matrix<double, 6, 1>::Constant(100.5));
    particles_stresses.emplace_back(
        Eigen::Matrix<double, 6, 1>::Constant(-112.5));
    particles_stresses.emplace_back(
        Eigen::Matrix<double, 6, 1>::Constant(0.46));

    // Dump initial stresses as an input file to be read
    std::ofstream file;
    file.open("particle-stresses-3d.txt");
    file << particles_stresses.size() << "\n";
    // Write particle coordinates
    for (const auto& stress : particles_stresses) {
      for (unsigned i = 0; i < stress.size(); ++i) file << stress[i] << "\t";
      file << "\n";
    }

    file.close();

    // Check read stresses
    SECTION("Check stresses") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read stresses from a non-existant file
      auto stresses = io_mesh->read_particles_stresses("stresses-missing.txt");
      // Check number of stresses
      REQUIRE(stresses.size() == 0);

      // Check stresses
      stresses = io_mesh->read_particles_stresses("particle-stresses-3d.txt");
      // Check number of particles
      REQUIRE(stresses.size() == particles_stresses.size());

      // Check stresses
      for (unsigned i = 0; i < particles_stresses.size(); ++i) {
        for (unsigned j = 0; j < particles_stresses.at(0).size(); ++j)
          REQUIRE(stresses.at(i)[j] ==
                  Approx(particles_stresses.at(i)[j]).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check particles scalar properties file") {
    // Particle scalar properties
    std::map<mpm::Index, double> particles_scalars;
    particles_scalars.emplace(std::make_pair(0, 10.5));
    particles_scalars.emplace(std::make_pair(1, -40.5));
    particles_scalars.emplace(std::make_pair(2, -60.5));
    particles_scalars.emplace(std::make_pair(3, 80.5));

    // Dump particle scalar properties as an input file to be read
    std::ofstream file;
    file.open("particles-scalars-3d.txt");
    // Write particle scalar properties
    for (const auto& scalars : particles_scalars) {
      file << scalars.first << "\t";
      file << scalars.second << "\n";
    }

    file.close();

    // Check read particles scalar properties file
    SECTION("Check read particles scalar properties file") {
      // Create a io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read particles scalar properties from a non-existant file
      auto read_particles_scalars =
          io_mesh->read_scalar_properties("particles-scalar-missing.txt");
      // Check number of particles scalar properties
      REQUIRE(read_particles_scalars.size() == 0);

      // Check particles scalar properties
      read_particles_scalars =
          io_mesh->read_scalar_properties("particles-scalars-3d.txt");

      // Check number of particles
      REQUIRE(read_particles_scalars.size() == particles_scalars.size());

      // Check particles scalar properties
      for (unsigned i = 0; i < particles_scalars.size(); ++i) {
        REQUIRE(std::get<1>(read_particles_scalars.at(i)) ==
                Approx(particles_scalars.at(i)).epsilon(Tolerance));
      }
    }
  }

  SECTION("Check math function file") {
    // Vector of math function entries
    std::array<std::vector<double>, 2> entries;

    // Populate the math function entries
    for (int i = 0; i < 4; ++i) {
      entries[0].push_back(0.3 * i);
      entries[1].push_back(2.6 * i);
    }

    // Dump the math entries as an input file to be read
    std::ofstream file;
    file.open("math-function-3d.csv");
    // Write math entries for x and fx
    for (int i = 0; i < 4; ++i)
      file << entries[0][i] << "," << entries[1][i] << "\n";

    file.close();

    // Check read math funciton file
    SECTION("Check math function entries") {
      // Create an io_mesh object
      auto io_mesh = std::make_unique<mpm::IOMeshAscii<dim>>();

      // Try to read math funciton entries from a non-existant file
      auto math_function_values =
          io_mesh->read_math_functions("math-function-missing.csv");
      // Check number of math function entries
      REQUIRE(math_function_values[0].size() == 0);
      REQUIRE(math_function_values[1].size() == 0);

      // Check math function reading
      math_function_values =
          io_mesh->read_math_functions("math-function-3d.csv");
      // Check number of entries from math function file
      REQUIRE(math_function_values[0].size() == entries[0].size());
      REQUIRE(math_function_values[1].size() == entries[1].size());
      REQUIRE(math_function_values[0].size() == math_function_values[1].size());

      // Check entry values
      for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < entries[0].size(); ++j)
          REQUIRE(math_function_values[i][j] ==
                  Approx(entries[i][j]).epsilon(Tolerance));
      }
    }
  }
}
