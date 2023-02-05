#pragma once

#include <Eigen/Eigen>
#include <cmath>
#include <functional>

namespace spkf
{
template <int Dx, int Du, int Dy>
class EKF
{
public:
    EKF(const Eigen::Matrix<float, Du, Du> &Q,
        const Eigen::Matrix<float, Dy, Dy> &R,
        const Eigen::Matrix<float, Dx, Dx> &P,
        const Eigen::Matrix<float, Dx, 1> &x,
        std::function<Eigen::Matrix<float, Dx, 1>(const Eigen::Matrix<float, Dx, 1> &,
                                                  const Eigen::Matrix<float, Du, 1> &,
                                                  const Eigen::Matrix<float, Dx, 1> &,
                                                  const float &dt)>
            F,
        std::function<Eigen::Matrix<float, Dy, 1>(const Eigen::Matrix<float, Dx, 1> &,
                                                  const Eigen::Matrix<float, Dy, 1> &)>
            G)

    {
        this->F = F;
        this->G = G;
        state_ = x;
        proc_noise_ = Eigen::Matrix<float, Dx, 1>::Zero();
        obs_noise_ = Eigen::Matrix<float, Dy, 1>::Zero();
        covar_ = P;
        proc_covar_ = Q;
        obs_covar_ = R;
        observation_ = Eigen::Matrix<float, Dy, 1>::Zero();
    }

    ~EKF()
    {
    }

    Eigen::Matrix<float, Dx, 1> Filter(Eigen::Matrix<float, Du, 1> u_input, Eigen::Matrix<float, Dy, 1> y_input, const float &dt)
    {
        this->dt = dt;
    }

private:
    void predict(const Eigen::Matrix<float, Du, 1> &control_k, const float dt, const Eigen::Matrix<float, Dx, 1> &proc_noise_k)
    {
        /*update process model*/
        proc_noise_ = proc_noise_k;

        // predict state mean using the process model
        process_(state_, control_k, dt);

        /*update state covariance*/
        process_covar_(covar_);
    }

    void innovate(const Eigen::Matrix<float, Dy, 1> &observation_k, const Eigen::Matrix<float, Dy, 1> &obs_noise_k)
    {
        /*update observation noise*/
        obs_noise_ = obs_noise_k;

        /*predict observation from predicted state*/
        observe_(observation_);

        /*innovation residual*/
        innovation_ = observation_k - observation_;

        /*update innovation covariance*/
        innovation_covar(inov_covar_);
    }

    void update()
    {
        /*compute kalman gain*/
        kalman_gain_update_(kalman_gain_);

        /*update state mean*/
        state_ += kalman_gain_ * innovation_;

        /*update state covariance*/
        update_covar_(covar_);
    }

    void kalman_gain_update_(Eigen::Matrix<float, Dx, Dy> &kalman_gain_k)
    {
        const auto &inov_covar_k = this->inov_covar();
        const auto &covar_k = this->covar();

        kalman_gain_k = inov_covar_k.transpose().fullPivHouseholderQr().solve((covar_k * proc_jacobian_G.transpose()).transpose()).transpose();
    }

    void update_covar_(Eigen::Matrix<float, Dx, Dx> &covar_k)
    {
        const auto &kalman_gain_k = this->kalman_gain();
        Eigen::Matrix<float, Dx, Dx> I = Eigen::Matrix<float, Dx, Dx>::Identity();
        covar_k = (I - kalman_gain_k * proc_jacobian_G) * covar_k;
    }

    void observe_(Eigen::Matrix<float, Dy, 1> &observation_k)
    {
        /*compute observation Jacobian at current state*/
        auto state_k = this->state();
        update_obs_jacobian_(state_k);

        /*predict measurement using the observation model*/
        auto obs_noise_k = this->obs_noise();
        observation_k = G(observation_k, obs_noise_k);
    }

    void innovation_covar(Eigen::Matrix<float, Dy, Dy> &inov_covar_k)
    {
        const auto &covar_k = this->covar();
        const auto &obs_covar_k = this->obs_covar();

        inov_covar_k = proc_jacobian_G * covar_k * proc_jacobian_G.transpose() + obs_covar_k;
    }

    void process_(Eigen::Matrix<float, Dx, 1> &state_k, const Eigen::Matrix<float, Du, 1> &control_k, const float dt)
    {
        /*compute process Jacobian at current state*/
        update_proc_jacobian_(state_k, control_k, dt);

        /*update state using the process model*/
        auto proc_noise_k = this->proc_noise();
        state_k = F(state_k, control_k, proc_noise_k, dt);
    }

    void process_covar_(Eigen::Matrix<float, Dx, Dx> &covar_k)
    {
        auto proc_covar_k = this->proc_covar();
        covar_k = proc_jacobian_F * covar_k * proc_jacobian_F.transpose() + proc_covar_k;
    }

    void update_proc_jacobian_(const Eigen::Matrix<float, Dx, 1> &state_k, const Eigen::Matrix<float, Du, 1> &control_k, const float dt)
    {
        Eigen::Matrix<float, Dx, 1> state_f;
        Eigen::Matrix<float, Dx, 1> state_b;

        for (int i = 0; i < Dx; ++i)
        {
            state_f = state_k;
            state_b = state_k;

            const float h_half = 0.5 * eps(state_f[i]);
            state_f[i] += h_half;
            state_b[i] -= h_half;

            /*pass through process model*/
            const auto &proc_noise_k = this->proc_noise();
            state_f = F(state_f, control_k, proc_noise_k, dt);
            state_b = F(state_b, control_k, proc_noise_k, dt);

            proc_jacobian_F.col(i) = (state_f - state_b) / (2.0 * h_half);
        }
    }

    void update_obs_jacobian_(const Eigen::Matrix<float, Dx, 1> &state_k)
    {
        Eigen::Matrix<float, Dy, 1> obs_f = Eigen::Matrix<float, Dy, 1>::Zero();
        Eigen::Matrix<float, Dy, 1> obs_b = Eigen::Matrix<float, Dy, 1>::Zero();

        Eigen::Matrix<float, Dx, 1> state_f;
        Eigen::Matrix<float, Dx, 1> state_b;

        for (int i = 0; i < Dy; ++i)
        {
            state_f = state_k;
            state_b = state_k;

            const float h_half = 0.5 * eps(state_f[i]);
            state_f[i] += h_half;
            state_b[i] -= h_half;

            const auto &obs_noise_k = this->obs_noise();
            obs_f = G(obs_f, obs_noise_k);
            obs_b = G(obs_b, obs_noise_k);

            proc_jacobian_G.col(i) = (obs_f - obs_b) / (2.0 * h_half);
        }
    }

    float eps(const float x)
    {
        const float sqrt_machine_eps = sqrt(std::numeric_limits<float>::epsilon());
        return (sqrt_machine_eps * std::max(std::abs(x), sqrt_machine_eps));
    }

    /*accessors*/
    inline const Eigen::Matrix<float, Dx, 1> &state() const
    {
        return state_;
    }
    inline const Eigen::Matrix<float, Dx, Dx> &covar() const
    {
        return covar_;
    }
    inline const Eigen::Matrix<float, Dy, 1> &observation() const
    {
        return observation_;
    }
    inline const Eigen::Matrix<float, Dy, 1> &innovation() const
    {
        return innovation_;
    }
    inline const Eigen::Matrix<float, Dx, Dx> &proc_covar() const
    {
        return proc_covar_;
    }
    inline const Eigen::Matrix<float, Dy, Dy> &obs_covar() const
    {
        return obs_covar_;
    }
    inline const Eigen::Matrix<float, Dx, Dy> &kalman_gain() const
    {
        return kalman_gain_;
    }
    inline const Eigen::Matrix<float, Dy, Dy> &inov_covar() const
    {
        return inov_covar_;
    }
    inline const Eigen::Matrix<float, Dx, 1> &proc_noise() const
    {
        return proc_noise_;
    }
    inline const Eigen::Matrix<float, Dy, 1> &obs_noise() const
    {
        return obs_noise_;
    }

private:
    float dt;

    /*process model*/
    std::function<Eigen::Matrix<float, Dx, 1>(const Eigen::Matrix<float, Dx, 1> &,
                                              const Eigen::Matrix<float, Du, 1> &,
                                              const Eigen::Matrix<float, Dx, 1> &,
                                              const float &dt)>
        F;

    /*observation model*/
    std::function<Eigen::Matrix<float, Dy, 1>(const Eigen::Matrix<float, Dx, 1> &,
                                              const Eigen::Matrix<float, Dy, 1> &)>
        G;

    Eigen::Matrix<float, Dx, 1> state_;
    Eigen::Matrix<float, Dx, Dx> covar_;
    Eigen::Matrix<float, Dy, 1> observation_;
    Eigen::Matrix<float, Dy, 1> innovation_;
    Eigen::Matrix<float, Dx, Dx> proc_covar_;
    Eigen::Matrix<float, Dy, Dy> obs_covar_;
    Eigen::Matrix<float, Dx, Dy> kalman_gain_;
    Eigen::Matrix<float, Dy, Dy> inov_covar_;
    Eigen::Matrix<float, Dx, 1> proc_noise_;
    Eigen::Matrix<float, Dy, 1> obs_noise_;
    Eigen::Matrix<float Dx, Dx> proc_jacobian_F;
    Eigen::Matrix<float, Dy, Dy> proc_jacobian_G;
};
} // namespace spkf