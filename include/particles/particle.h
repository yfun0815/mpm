#ifndef MPM_PARTICLE_H_
#define MPM_PARTICLE_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "cell.h"
#include "logger.h"
#include "math_utility.h"
#include "particle_base.h"

namespace mpm {

//! Particle class
//! \brief Base class that stores the information about particles
//! \details Particle class: id_ and coordinates.
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Particle : public ParticleBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define DOFs
  static const unsigned Tdof = (Tdim == 1) ? 1 : 3 * (Tdim - 1);

  //! Construct a particle with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  Particle(Index id, const VectorDim& coord);

  //! Construct a particle with id, coordinates and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  Particle(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~Particle() override{};

  //! Delete copy constructor
  Particle(const Particle<Tdim>&) = delete;

  //! Delete assignment operator
  Particle& operator=(const Particle<Tdim>&) = delete;

  //! Initialise particle from POD data
  //! \param[in] particle POD data of particle
  //! \retval status Status of reading POD particle
  bool initialise_particle(PODParticle& particle) override;

  //! Initialise particle POD data and material
  //! \param[in] particle POD data of particle
  //! \param[in] materials Material associated with the particle arranged in a
  //! vector
  //! \retval status Status of reading POD particle
  bool initialise_particle(
      PODParticle& particle,
      const std::vector<std::shared_ptr<Material<Tdim>>>& materials) override;

  //! Return particle data as POD
  //! \retval particle POD of the particle
  std::shared_ptr<void> pod() const override;

  //! Initialise properties
  void initialise() override;

  //! Compute reference coordinates in a cell
  bool compute_reference_location() noexcept override;

  //! Return reference location
  VectorDim reference_location() const override { return xi_; }

  //! Assign a cell to particle
  //! If point is in new cell, assign new cell and remove particle id from old
  //! cell. If point can't be found in the new cell, check if particle is still
  //! valid in the old cell, if it is leave it as is. If not, set cell as null
  //! \param[in] cellptr Pointer to a cell
  bool assign_cell(const std::shared_ptr<Cell<Tdim>>& cellptr) override;

  //! Assign a cell to particle
  //! If point is in new cell, assign new cell and remove particle id from old
  //! cell. If point can't be found in the new cell, check if particle is still
  //! valid in the old cell, if it is leave it as is. If not, set cell as null
  //! \param[in] cellptr Pointer to a cell
  //! \param[in] xi Local coordinates of the point in reference cell
  bool assign_cell_xi(const std::shared_ptr<Cell<Tdim>>& cellptr,
                      const Eigen::Matrix<double, Tdim, 1>& xi) override;

  //! Assign cell id
  //! \param[in] id Cell id
  bool assign_cell_id(Index id) override;

  //! Return cell id
  Index cell_id() const override { return cell_id_; }

  //! Return cell ptr status
  bool cell_ptr() const override { return cell_ != nullptr; }

  //! Remove cell associated with the particle
  void remove_cell() override;

  //! Compute shape functions of a particle, based on local coordinates
  void compute_shapefn() noexcept override;

  //! Assign volume
  //! \param[in] volume Volume of particle
  bool assign_volume(double volume) override;

  //! Return volume
  double volume() const override { return volume_; }

  //! Return the approximate particle diameter
  double diameter() const override;

  //! Return size of particle in natural coordinates
  VectorDim natural_size() const override { return natural_size_; }

  //! Compute volume as cell volume / nparticles
  void compute_volume() noexcept override;

  //! Update volume based on centre volumetric strain rate
  void update_volume() noexcept override;

  //! Return mass density
  double mass_density() const override { return mass_density_; }

  //! Compute mass as volume * density
  void compute_mass() noexcept override;

  //! Map particle mass and momentum to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  void map_mass_momentum_to_nodes(
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::FLIP) noexcept override;

  //! Map multimaterial properties to nodes
  void map_multimaterial_mass_momentum_to_nodes() noexcept override;

  //! Map multimaterial displacements to nodes
  void map_multimaterial_displacements_to_nodes() noexcept override;

  //! Map multimaterial domain gradients to nodes
  void map_multimaterial_domain_gradients_to_nodes() noexcept override;

  // ! Map linear elastic wave velocities to nodes
  void map_wave_velocities_to_nodes() noexcept override;

  //! Assign nodal mass to particles
  //! \param[in] mass Mass from the particles in a cell
  //! \retval status Assignment status
  void assign_mass(double mass) override { mass_ = mass; }

  //! Return mass of the particles
  double mass() const override { return mass_; }

  //! Assign material
  //! \param[in] material Pointer to a material
  //! \param[in] phase Index to indicate phase
  bool assign_material(const std::shared_ptr<Material<Tdim>>& material,
                       unsigned phase = mpm::ParticlePhase::Solid) override;

  //! Compute strain
  //! \param[in] dt Analysis time step
  void compute_strain(double dt) noexcept override;

  //! Return strain of the particle
  Eigen::Matrix<double, 6, 1> strain() const override { return strain_; }

  //! Return strain rate of the particle
  Eigen::Matrix<double, 6, 1> strain_rate() const override {
    return strain_rate_;
  };

  //! Return dvolumetric strain of centroid
  //! \retval dvolumetric strain at centroid
  double dvolumetric_strain() const override { return dvolumetric_strain_; }

  //! Assign deformation gradient increment
  void assign_deformation_gradient_increment(
      Eigen::Matrix<double, 3, 3> F_inc) noexcept override {
    deformation_gradient_increment_ = F_inc;
  }

  //! Assign deformation gradient
  void assign_deformation_gradient(
      Eigen::Matrix<double, 3, 3> F) noexcept override {
    deformation_gradient_ = F;
  }

  //! Return deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment() const override {
    return deformation_gradient_increment_;
  }

  //! Return Deformation gradient
  Eigen::Matrix<double, 3, 3> deformation_gradient() const override {
    return deformation_gradient_;
  }

  //! Update deformation gradient increment using displacement (for implicit
  //! schemes)
  void update_deformation_gradient_increment() noexcept override;

  //! Update deformation gradient increment using velocity (for explicit
  //! schemes)
  //! \param[in] dt Analysis time step
  void update_deformation_gradient_increment(double dt) noexcept override;

  //! Update deformation gradient provided that the deformation gradient
  //! increment exists
  void update_deformation_gradient() noexcept override;

  //! Initial stress
  //! \param[in] stress Initial sress
  void initial_stress(const Eigen::Matrix<double, 6, 1>& stress) override {
    this->stress_ = stress;
    this->previous_stress_ = stress;
  }

  //! Compute stress
  //! \param[in] dt Analysis time step
  void compute_stress(double dt) noexcept override;

  //! Return stress of the particle
  Eigen::Matrix<double, 6, 1> stress() const override { return stress_; }

  //! Map body force
  //! \param[in] pgravity Gravity of a particle
  void map_body_force(const VectorDim& pgravity) noexcept override;

  //! Map internal force
  inline void map_internal_force() noexcept override;

  //! Assign velocity to the particle
  //! \param[in] velocity A vector of particle velocity
  //! \retval status Assignment status
  bool assign_velocity(const VectorDim& velocity) override;

  //! Return velocity of the particle
  VectorDim velocity() const override { return velocity_; }

  //! Return displacement of the particle
  VectorDim displacement() const override { return displacement_; }

  //! Assign traction to the particle
  //! \param[in] direction Index corresponding to the direction of traction
  //! \param[in] traction Particle traction in specified direction
  //! \retval status Assignment status
  bool assign_traction(unsigned direction, double traction) override;

  //! Return traction of the particle
  VectorDim traction() const override { return traction_; }

  //! Map traction force
  void map_traction_force() noexcept override;

  //! Compute updated position of the particle
  //! \param[in] dt Analysis time step
  //! \param[in] velocity_update Method to update particle velocity
  //! \param[in] blending_ratio FLIP-PIC Blending ratio
  void compute_updated_position(
      double dt,
      mpm::VelocityUpdate velocity_update = mpm::VelocityUpdate::FLIP,
      double blending_ratio = 1.0) noexcept override;

  //! Assign material history variables
  //! \param[in] state_vars State variables
  //! \param[in] material Material associated with the particle
  //! \param[in] phase Index to indicate material phase
  //! \retval status Status of assigning material state variables
  bool assign_material_state_vars(
      const mpm::dense_map& state_vars,
      const std::shared_ptr<mpm::Material<Tdim>>& material,
      unsigned phase = mpm::ParticlePhase::Solid) override;

  //! Assign a state variable
  //! \param[in] var State variable
  //! \param[in] value State variable to be assigned
  //! \param[in] phase Index to indicate phase
  void assign_state_variable(
      const std::string& var, double value,
      unsigned phase = mpm::ParticlePhase::Solid) override;

  //! Return a state variable
  //! \param[in] var State variable
  //! \param[in] phase Index to indicate phase
  //! \retval Quantity of the state history variable
  double state_variable(
      const std::string& var,
      unsigned phase = mpm::ParticlePhase::Solid) const override {
    return (phase < state_variables_.size() &&
            state_variables_[phase].find(var) != state_variables_[phase].end())
               ? state_variables_[phase].at(var)
               : std::numeric_limits<double>::quiet_NaN();
  }

  //! Map particle pressure to nodes
  bool map_pressure_to_nodes(
      unsigned phase = mpm::ParticlePhase::Solid) noexcept override;

  //! Compute pressure smoothing of the particle based on nodal pressure
  //! $$\hat{p}_p = \sum_{i = 1}^{n_n} N_i(x_p) p_i$$
  bool compute_pressure_smoothing(
      unsigned phase = mpm::ParticlePhase::Solid) noexcept override;

  //! Assign a state variable
  //! \param[in] value Particle pressure to be assigned
  //! \param[in] phase Index to indicate phase
  void assign_pressure(double pressure,
                       unsigned phase = mpm::ParticlePhase::Solid) override {
    this->assign_state_variable("pressure", pressure, phase);
  }

  //! Return pressure of the particles
  //! \param[in] phase Index to indicate phase
  double pressure(unsigned phase = mpm::ParticlePhase::Solid) const override {
    return this->state_variable("pressure", phase);
  }

  //! Return scalar data of particles
  //! \param[in] property Property string
  //! \retval data Scalar data of particle property
  inline double scalar_data(const std::string& property) const override;

  //! Return vector data of particles
  //! \param[in] property Property string
  //! \retval data Vector data of particle property
  inline VectorDim vector_data(const std::string& property) const override;

  //! Return tensor data of particles
  //! \param[in] property Property string
  //! \retval data Tensor data of particle property
  inline Eigen::VectorXd tensor_data(
      const std::string& property) const override;

  //! Apply particle velocity constraints
  //! \param[in] dir Direction of particle velocity constraint
  //! \param[in] velocity Applied particle velocity constraint
  void apply_particle_velocity_constraints(unsigned dir,
                                           double velocity) override;

  //! Assign material id of this particle to nodes
  void append_material_id_to_nodes() const override;

  //! Assign free surface
  void assign_free_surface(bool free_surface) override {
    free_surface_ = free_surface;
  };

  //! Return free surface bool
  bool free_surface() const override { return free_surface_; };

  //! Compute free surface in particle level by density ratio comparison
  //! \param[in] density_ratio_tolerance Tolerance of density ratio comparison.
  //! Default value is set to be 0.65, which is derived from a 3D case where at
  //! one side the cell is fully occupied by particles and the other side the
  //! cell is empty. See (Hamad, 2015).
  //! \retval status Status of compute_free_surface
  bool compute_free_surface_by_density(
      double density_ratio_tolerance = 0.65) override;

  //! Assign normal vector
  void assign_normal(const VectorDim& normal) override { normal_ = normal; };

  //! Return normal vector
  VectorDim normal() const override { return normal_; };

  //! Return the number of neighbour particles
  unsigned nneighbours() const override { return neighbours_.size(); };

  //! Assign neighbour particles
  //! \param[in] neighbours set of id of the neighbouring particles
  //! \retval insertion_status Return the successful addition of a node
  void assign_neighbours(const std::vector<mpm::Index>& neighbours) override;

  //! Return neighbour ids
  std::vector<mpm::Index> neighbours() const override { return neighbours_; }

  //! Type of particle
  std::string type() const override { return (Tdim == 2) ? "P2D" : "P3D"; }

  //! Serialize
  //! \retval buffer Serialized buffer data
  std::vector<uint8_t> serialize() override;

  //! Deserialize
  //! \param[in] buffer Serialized buffer data
  //! \param[in] material Particle material pointers
  void deserialize(
      const std::vector<uint8_t>& buffer,
      std::vector<std::shared_ptr<mpm::Material<Tdim>>>& materials) override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Map particle mass, momentum and inertia to nodes
  //! \ingroup Implicit
  void map_mass_momentum_inertia_to_nodes() noexcept override;

  //! Map inertial force
  //! \ingroup Implicit
  void map_inertial_force() noexcept override;

  //! Assign acceleration to the particle (used for test)
  //! \ingroup Implicit
  //! \param[in] acceleration A vector of particle acceleration
  //! \retval status Assignment status
  bool assign_acceleration(const VectorDim& acceleration) override;

  //! Return acceleration of the particle
  //! \ingroup Implicit
  VectorDim acceleration() const override { return acceleration_; }

  //! Map mass and material stiffness matrix to cell (used in equilibrium
  //! equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt parameter beta of Newmark scheme
  //! \param[in] quasi_static Boolean of quasi-static analysis
  inline bool map_stiffness_matrix_to_cell(double newmark_beta, double dt,
                                           bool quasi_static) override;

  //! Reduce constitutive relations matrix depending on the dimension
  //! \ingroup Implicit
  //! \param[in] dmatrix Constitutive relations matrix in 3D
  //! \retval reduced_dmatrix Reduced constitutive relation matrix for spatial
  //! dimension
  inline Eigen::MatrixXd reduce_dmatrix(
      const Eigen::MatrixXd& dmatrix) noexcept override;

  //! Compute B matrix of a particle, based on local coordinates
  inline Eigen::MatrixXd compute_bmatrix() noexcept override;

  //! Compute strain and volume using nodal displacement
  //! \ingroup Implicit
  void compute_strain_volume_newmark() noexcept override;

  //! Compute stress using implicit updating scheme
  //! \ingroup Implicit
  //! \param[in] dt Analysis time step
  void compute_stress_newmark(double dt) noexcept override;

  //! Return stress at the previous time step of the particle
  //! \ingroup Implicit
  Eigen::Matrix<double, 6, 1> previous_stress() const override {
    return previous_stress_;
  }

  //! Compute updated position of the particle by Newmark scheme
  //! \ingroup Implicit
  //! \param[in] dt Analysis time step
  void compute_updated_position_newmark(double dt) noexcept override;

  //! Update stress and strain after convergence of Newton-Raphson iteration
  //! \ingroup Implicit
  //! \param[in] dt Analysis time step
  void update_stress_strain(double dt) noexcept override;

  //! Function to reinitialise consitutive law to be run at the beginning of
  //! each time step
  //! \ingroup Implicit
  //! \param[in] dt Analysis time step
  void initialise_constitutive_law(double dt) noexcept override;
  /**@}*/

  void reset_displacement() override { displacement_.setZero(); }

 protected:
  //! Initialise particle material container
  //! \details This function allocate memory and initialise the material related
  //! containers according to the particle phase, i.e. solid or fluid particle
  //! has phase_size = 1, whereas two-phase (solid-fluid) or three-phase
  //! (solid-water-air) particle have phase_size = 2 and 3, respectively.
  //! \param[in] phase_size The material phase size
  void initialise_material(unsigned phase_size = 1);

  //! Compute strain rate
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain rate at particle inside a cell
  virtual inline Eigen::Matrix<double, 6, 1> compute_strain_rate(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

  //! Compute pack size
  //! \retval pack size of serialized object
  virtual int compute_pack_size() const;

  //! Compute deformation gradient increment using nodal velocity
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \param[in] dt time increment
  //! \retval deformaton gradient increment at particle inside a cell
  inline Eigen::Matrix<double, 3, 3> compute_deformation_gradient_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept;

  //! Compute velocity gradient
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval velocity gradient increment at particle inside a cell
  inline Eigen::Matrix<double, Tdim, Tdim> compute_velocity_gradient(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute strain increment
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain increment at particle inside a cell
  virtual inline Eigen::Matrix<double, 6, 1> compute_strain_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

  //! Compute deformation gradient increment using nodal displacement
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval deformaton gradient increment at particle inside a cell
  inline Eigen::Matrix<double, 3, 3> compute_deformation_gradient_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

  //! Map material stiffness matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  inline bool map_material_stiffness_matrix_to_cell();

  //! Map mass matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt parameter beta of Newmark scheme
  inline bool map_mass_matrix_to_cell(double newmark_beta, double dt);
  /**@}*/

  /**
   * \defgroup AdvancedMapping Functions dealing with advance mapping scheme of
   * MPM
   */
  /**@{*/
  //! Return mapping matrix
  //! \ingroup AdvancedMapping
  Eigen::MatrixXd mapping_matrix() const override { return mapping_matrix_; }

  //! Map particle mass and momentum to nodes for affine transformation
  //! \ingroup AdvancedMapping
  virtual void map_mass_momentum_to_nodes_affine() noexcept;

  //! Map particle mass and momentum to nodes for approximate taylor expansion
  //! \ingroup AdvancedMapping
  virtual void map_mass_momentum_to_nodes_taylor() noexcept;

  //! Compute updated position of the particle assuming FLIP scheme
  //! \ingroup AdvancedMapping
  //! \param[in] dt Analysis time step
  //! \param[in] blending_ratio FLIP-PIC Blending ratio
  void compute_updated_position_flip(double dt,
                                     double blending_ratio = 1.0) noexcept;

  //! Compute updated position of the particle assuming PIC scheme
  //! \ingroup AdvancedMapping
  //! \param[in] dt Analysis time step
  void compute_updated_position_pic(double dt) noexcept;

  //! Compute updated position of the particle assuming ASFLIP scheme
  //! \ingroup AdvancedMapping
  //! \param[in] dt Analysis time step
  //! \param[in] blending_ratio FLIP-PIC Blending ratio
  void compute_updated_position_asflip(double dt,
                                       double blending_ratio = 1.0) noexcept;

  //! Compute updated position of the particle assuming APIC scheme
  //! \ingroup AdvancedMapping
  //! \param[in] dt Analysis time step
  void compute_updated_position_apic(double dt) noexcept;

  //! Compute updated position of the particle assuming TPIC scheme
  //! \ingroup AdvancedMapping
  //! \param[in] dt Analysis time step
  void compute_updated_position_tpic(double dt) noexcept;

  //! Compute Affine B-Matrix for all the affine scheme
  //! \ingroup AdvancedMapping
  //! \param[in] shapefn Shape function
  //! \param[in] phase Index to indicate phase
  //! \retval velocity gradient increment at particle inside a cell
  inline Eigen::Matrix<double, Tdim, Tdim> compute_affine_mapping_matrix(
      const Eigen::MatrixXd& shapefn, unsigned phase) noexcept;

  //! Compute ASFLIP beta parameter
  //! \ingroup AdvancedMapping
  //! \param[in] dt time increment
  inline double compute_asflip_beta(double dt) noexcept;

  /**@}*/

  //! particle id
  using ParticleBase<Tdim>::id_;
  //! coordinates
  using ParticleBase<Tdim>::coordinates_;
  //! Reference coordinates (in a cell)
  using ParticleBase<Tdim>::xi_;
  //! Cell
  using ParticleBase<Tdim>::cell_;
  //! Cell id
  using ParticleBase<Tdim>::cell_id_;
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! Status
  using ParticleBase<Tdim>::status_;
  //! Material
  using ParticleBase<Tdim>::material_;
  //! Material id
  using ParticleBase<Tdim>::material_id_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Neighbour particles
  using ParticleBase<Tdim>::neighbours_;
  //! Volumetric mass density (mass / volume)
  double mass_density_{0.};
  //! Mass
  double mass_{0.};
  //! Volume
  double volume_{0.};
  //! Size of particle
  double size_;
  //! Size of particle in natural coordinates
  Eigen::Matrix<double, Tdim, 1> natural_size_;
  //! Stresses
  Eigen::Matrix<double, 6, 1> stress_;
  //! Strains
  Eigen::Matrix<double, 6, 1> strain_;
  //! dvolumetric strain
  double dvolumetric_strain_{0.};
  //! Strain rate
  Eigen::Matrix<double, 6, 1> strain_rate_;
  //! dstrains
  Eigen::Matrix<double, 6, 1> dstrain_;
  //! Velocity
  Eigen::Matrix<double, Tdim, 1> velocity_;
  //! Displacement
  Eigen::Matrix<double, Tdim, 1> displacement_;
  //! Particle velocity constraints
  std::map<unsigned, double> particle_velocity_constraints_;
  //! Free surface
  bool free_surface_{false};
  //! Free surface
  Eigen::Matrix<double, Tdim, 1> normal_;
  //! Set traction
  bool set_traction_{false};
  //! Surface Traction (given as a stress; force/area)
  Eigen::Matrix<double, Tdim, 1> traction_;
  //! Shape functions
  Eigen::VectorXd shapefn_;
  //! dN/dX
  Eigen::MatrixXd dn_dx_;
  //! dN/dX at cell centroid
  Eigen::MatrixXd dn_dx_centroid_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Map of scalar properties
  tsl::robin_map<std::string, std::function<double()>> scalar_properties_;
  //! Map of vector properties
  tsl::robin_map<std::string, std::function<VectorDim()>> vector_properties_;
  //! Map of tensor properties
  tsl::robin_map<std::string, std::function<Eigen::VectorXd()>>
      tensor_properties_;
  //! Pack size
  unsigned pack_size_{0};
  //! Mapping matrix for advance mapping schemes
  Eigen::MatrixXd mapping_matrix_;

  /**
   * \defgroup ImplicitVariables Variables dealing with implicit MPM
   */
  /**@{*/
  //! Acceleration
  Eigen::Matrix<double, Tdim, 1> acceleration_;
  //! Stresses at the last time step
  Eigen::Matrix<double, 6, 1> previous_stress_;
  //! Constitutive Tangent Matrix (dynamic allocation only for implicit scheme)
  Eigen::MatrixXd constitutive_matrix_;
  /**@}*/

  /**
   * \defgroup FiniteStrainVariables Variables for finite strain formulation
   */
  /**@{*/
  //! Deformation gradient
  Eigen::Matrix<double, 3, 3> deformation_gradient_;
  //! Deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment_;
  /**@}*/

};  // Particle class
}  // namespace mpm

#include "particle.tcc"
#include "particle_implicit.tcc"

#endif  // MPM_PARTICLE_H__
