#include "concrete_damage_plasticity.h"
#include <cmath>
#include <algorithm>

ConcreteDamagePlasticity::ConcreteDamagePlasticity(const std::map<std::string, double>& material_properties) {
    ft_ = material_properties.at("tensile_strength");
    fc_ = material_properties.at("compressive_strength");
    alpha_t_ = material_properties.at("alpha_t");
    alpha_c_ = material_properties.at("alpha_c");
    ftr_ = material_properties.at("ftr");
    fcr_ = material_properties.at("fcr");
    epsilon_p_t0_ = material_properties.at("epsilon_p_t0");
    epsilon_p_c0_ = material_properties.at("epsilon_p_c0");

    // 初始化损伤变量
    damage_t_ = 0.0;
    damage_c_ = 0.0;
}

Eigen::Matrix3d ConcreteDamagePlasticity::compute_elasto_plastic_tensor(
    const Eigen::VectorXd& stress, std::map<std::string, double>* state_vars) {

    // 计算应力张量的主值
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver;
    Eigen::Matrix3d stress_tensor;
    stress_tensor << stress(0), stress(3), stress(5),
                     stress(3), stress(1), stress(4),
                     stress(5), stress(4), stress(2);
    eig_solver.compute(stress_tensor);
    double max_principal = eig_solver.eigenvalues().maxCoeff();

    // 更新塑性应变和损伤
    double& ep_t = (*state_vars)["plastic_strain_t"];
    double& ep_c = (*state_vars)["plastic_strain_c"];
    
    if (max_principal > 0) {
        // 张拉
        ep_t += stress.norm();  // 简化处理，应为塑性分量
        if (ep_t > epsilon_p_t0_) {
            double dt = 1. - (ftr_ / ft_) * (epsilon_p_t0_ / ep_t) *
                              std::exp(-alpha_t_ * (ep_t - epsilon_p_t0_));
            (*state_vars)["damage_t"] = std::min(dt, 0.99);  // 限制最大值
        }
    } else {
        // 压缩
        ep_c += stress.norm();
        if (ep_c > epsilon_p_c0_) {
            double dc = 1. - (fcr_ / fc_) * (epsilon_p_c0_ / ep_c) *
                              std::exp(-alpha_c_ * (ep_c - epsilon_p_c0_));
            (*state_vars)["damage_c"] = std::min(dc, 0.99);
        }
    }

    // 选择最大损伤变量
    double damage = std::max((*state_vars)["damage_t"], (*state_vars)["damage_c"]);

    // 计算弹性张量并乘以损伤因子
    Eigen::Matrix3d D_elastic = Eigen::Matrix3d::Identity();  // 需要根据实际情况计算弹性张量
    D_elastic *= (1. - damage);  // 损伤因子的应用

    return D_elastic;
}
