#ifndef MPM_MATERIAL_LINEAR_ELASTIC_H_
#define MPM_MATERIAL_LINEAR_ELASTIC_H_

#include <limits>

#include "Eigen/Dense"
#include <unsupported/Eigen/MatrixFunctions>

#include "material.h"

namespace mpm {

//! LinearElastic class
//! \brief Linear Elastic material model
//! \details LinearElastic class stresses and strains
//! \tparam Tdim Dimension
template <unsigned Tdim>
class LinearElastic : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id
  //! \param[in] material_properties Material properties
  LinearElastic(unsigned id, const Json& material_properties);

  //! Destructor
  ~LinearElastic() override{};

  //! Delete copy constructor
  LinearElastic(const LinearElastic&) = delete;

  //! Delete assignement operator
  LinearElastic& operator=(const LinearElastic&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables(double y) override {
    mpm::dense_map state_vars;
    return state_vars;
  }

  //! State variables
  std::vector<std::string> state_variables() const override { return {}; }

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;

  //! Compute consistent tangent matrix
  //! \param[in] stress Updated stress
  //! \param[in] prev_stress Stress at the current step
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_consistent_tangent_matrix(const Vector6d& stress,
                                              const Vector6d& prev_stress,
                                              const Vector6d& dstrain,
                                              const ParticleBase<Tdim>* ptr,
                                              mpm::dense_map* state_vars,
                                              double dt) override;

 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;
  //! Objective stress rate
  mpm::StressRate stress_rate_{mpm::StressRate::None};

  //! Compute stress using objective algorithm assuming Jaumann rate
  //! \param[in] stress Stress (Voigt)
  //! \param[in] dstrain Strain (Voigt)
  //! \param[in] de Elastic constitutive tensor (Voigt)
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress with Jaumann rate
  virtual Vector6d compute_jaumann_stress(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const Matrix6x6& de,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars);

  //! Compute stress using objective algorithm assuming Green-Naghdi rate
  //! \param[in] stress Stress (Voigt)
  //! \param[in] dstrain Strain (Voigt)
  //! \param[in] de Elastic constitutive tensor (Voigt)
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress with Green-Naghdi rate
  virtual Vector6d compute_green_naghdi_stress(const Vector6d& stress,
                                               const Vector6d& dstrain,
                                               const Matrix6x6& de,
                                               const ParticleBase<Tdim>* ptr,
                                               mpm::dense_map* state_vars);

 private:
  //! Compute elastic tensor
  bool compute_elastic_tensor();

 private:
  //! Elastic stiffness matrix
  Matrix6x6 de_;
  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Youngs modulus
  double youngs_modulus_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! Bulk modulus
  double bulk_modulus_{std::numeric_limits<double>::max()};
  //! Compressional Wave Velocity
  double vp_{std::numeric_limits<double>::max()};
  //! Shear Wave Velocity
  double vs_{std::numeric_limits<double>::max()};
};  // LinearElastic class
}  // namespace mpm

#include "linear_elastic.tcc"

#endif  // MPM_MATERIAL_LINEAR_ELASTIC_H_