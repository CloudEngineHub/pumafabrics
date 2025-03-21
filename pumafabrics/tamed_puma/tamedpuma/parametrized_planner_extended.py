import logging
import casadi as ca
import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from forwardkinematics.fksCommon.fk import ForwardKinematics
class ParameterizedFabricPlannerExtended(ParameterizedFabricPlanner):
    def __init__(self, dof: int, forward_kinematics: ForwardKinematics, time_step: float, **kwargs):
        super().__init__(dof=dof, forward_kinematics=forward_kinematics, **kwargs)
        self._time_step = time_step

    ### ---- Additional functions if you would like to analyse geometry groups separately --- ####
    def add_speed_control(self, bool_speed_control=True):
        """
        Add speed control to the forced geometry.
        Theory can be found in chapter 8 of Ratliff, Optimization Fabrics
        :return: xddot
        """
        try:
            xddot_nospeed = self._forced_geometry._xddot
        except AttributeError:
            logging.warn("No forcing term, using pure geoemtry with energization.")
            xddot_nospeed = self._execution_geometry._xddot

        if bool_speed_control == True:
            try:
                eta = self._damper.substitute_eta()
                # Execution alpha: interpolation between alpha^0_ex (not forced) and alpha^psi_ex (forced), (Proposition 8.1):
                a_ex = (
                        eta * self._execution_geometry._alpha
                        + (1 - eta) * self._forced_speed_controlled_geometry._alpha
                )
                beta_subst = self._damper.substitute_beta(-a_ex, -self._geometry._alpha)

                # Ratliff, Optimization fabric, equation 89:
                xddot = self._forced_geometry._xddot  - (a_ex + beta_subst) * (
                        self._geometry.xdot()
                        - ca.mtimes(self._forced_geometry.Minv(), self._target_velocity)
                )
            except AttributeError:
                # If there is no forcing term, equation 89 simplifies to the following
                logging.warn("No forcing term, using pure geoemtry with energization.")
                self._geometry.concretize()
                #xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
                xddot = self._execution_geometry._xddot - self._execution_geometry._alpha * self._geometry._vars.velocity_variable()
        else: #without speed control (with and without forcing)
            xddot = xddot_nospeed

        diff_xddot_speed = xddot - xddot_nospeed
        return xddot, diff_xddot_speed

    def concretize_extensive(self, mode='acc', time_step=None, extensive_concretize=False, bool_speed_control=True):
        self._mode = mode
        self._time_step = time_step
        if mode == 'vel':
            if not time_step:
                raise Exception("No time step passed in velocity mode.")

        xddot, diff_xddot_split = self.add_speed_control(bool_speed_control=bool_speed_control)

        if extensive_concretize == True:
            self.extensive_concretize_geometries(speed_control_addition = diff_xddot_split)

        if mode == 'acc':
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action": xddot}
            )
        elif mode == 'vel':
            action = self._geometry.xdot() + time_step * xddot
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action": action}
            )

    def extensive_concretize_geometries(self, speed_control_addition):
        """
        For some applications, it is useful to retrieve the function of M, f and actions of the
        (1) all avoidance geometries, (2) attractor geometries, (3) full geometry, separately.
        These functions can then be called numerically later on.
        :return: functions of different geometries
        """
        if self._mode == 'vel':
            if not self._time_step:
                raise Exception("No time step passed in velocity mode.")
        try:
            self._geometry.concretize()
            self._attractor_geometry = self._forced_geometry
            self._attractor_geometry.concretize()
            self._forced_geometry.concretize()

            # full geometry
            M_forced = self._forced_geometry._funs._expressions["M"]
            f_forced = self._forced_geometry._funs._expressions["f"]
            xddot_forced = self._forced_geometry._xddot

            #only avoidance components
            M_avoidance = self._geometry._M
            f_avoidance = self._geometry._funs._expressions["f"]
            xddot_avoidance = self._geometry._xddot

            #attractor geometry (goal reaching)
            M_attractor = self._attractor_geometry._M
            f_attractor = self._attractor_geometry._funs._expressions["f"]
            xddot_attractor = self._attractor_geometry._xddot
        except AttributeError:
            logging.warn("No forcing term, using pure geometry with energization.")
            self._geometry.concretize()
            M_avoidance = self._geometry._M
            f_avoidance = self._geometry._funs._expressions["f"]
            xddot_avoidance = self._geometry._xddot

        try:
            if self._mode == 'acc':
                action = xddot_forced
                action_attractor = xddot_attractor
                action_avoidance = xddot_avoidance
            elif self._mode == 'vel':
                action = self._geometry.xdot() + self._time_step * xddot_forced #todo: replace by _forced_geometry?
                action_attractor = self._attractor_geometry.xdot() + self._time_step * xddot_attractor
                action_avoidance = self._geometry.xdot() + self._time_step * xddot_avoidance

            self._funs_full = CasadiFunctionWrapper(
                "funs_full", self.variables, {"action": action, "M":M_forced, "f":f_forced, "xddot_speed":speed_control_addition}
            )
            self._funs_attractor = CasadiFunctionWrapper(
                "funs_attractor", self.variables, {"action": action_attractor, "M":M_attractor, "f":f_attractor, "xddot_speed":speed_control_addition}
            )
            self._funs_avoidance = CasadiFunctionWrapper(
                "funs_attractor", self.variables, {"action": action_avoidance, "M":M_avoidance, "f":f_avoidance, "xddot_speed":speed_control_addition}
            )
        except:
            logging.warn("No forcing term, using pure geometry with energization.")

            if self._mode == 'acc':
                action_avoidance = xddot_avoidance
            elif self._mode == 'vel':
                action_avoidance = self._geometry.xdot() + self._time_step * xddot_avoidance

            self._funs_avoidance = CasadiFunctionWrapper(
                "funs_attractor", self.variables, {"action": action_avoidance, "M":M_avoidance, "f":f_avoidance, "xddot_speed":speed_control_addition}
            )

    def compute_M_f_action(self, **kwargs):
        try:
            evaluations = self._funs_full.evaluate(**kwargs)
            M, f, action, xddot_speed = self.get_M_f_action(evaluations=evaluations)
        except:
            print("If you have no goal component, use compute_M_f_action_avoidance instead of this function")
        return M, f, action, xddot_speed

    def compute_M_f_action_avoidance(self, **kwargs):
        evaluations = self._funs_avoidance.evaluate(**kwargs)
        M, f, action, xddot_speed = self.get_M_f_action(evaluations=evaluations)
        return M, f, action, xddot_speed

    def compute_M_f_action_attractor(self, **kwargs):
        evaluations = self._funs_attractor.evaluate(**kwargs)
        M, f, action, xddot_speed = self.get_M_f_action(evaluations=evaluations)
        return M, f, action, xddot_speed

    def get_M_f_action(self, evaluations):
        M = evaluations["M"]
        f = evaluations["f"]
        action = evaluations["action"]
        xddot_speed = evaluations["xddot_speed"]
        return M, f, action, xddot_speed

    def Minv(self, M, eps=1e-6):
        """
        This is a copy of Minv as in spec.py, but where M and epsilon can be given as inputs.
        :param M: M matrix of a spec.
        :param eps: small number, 1e-6 to avoid weird pseudoinverse
        :return: pseudinverse(M)
        """
        logging.warning("Casadi pseudo inverse is used in spec")
        return ca.pinv(M + np.identity(len(M)) * eps)