import spatial_casadi as sc
import casadi as ca
import copy
class SymbolicKinScaling:
    def __init__(self):
        dt = 0.01

    def forward_kinematics_symbolic(self, q, end_link_name="iiwa_link_7", fk=None):
        x_fk = fk.fk(q=q, parent_link="iiwa_link_0", child_link=end_link_name, positionOnly=False)
        pos = x_fk[:3, 3]
        rot_matrix = x_fk[:3, :3]
        quat = self.symbolic_rot_matrix_to_quaternions(rot_matrix=rot_matrix)
        x_pose = ca.vcat((pos, quat))
        return x_pose

    def symbolic_rot_matrix_to_quaternions(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        quatern = r.as_quat()
        return quatern

    def normalize_pose_to_NN(self, x_t, translation_gpu, offset_orientation):
        # transformations
        x_gpu, x_cpu = self.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu)
        # quaternion offset
        x_gpu[0][3:7] = self.system_quat_to_NN(quat=x_cpu[3:7], offset=offset_orientation)
        return x_gpu

    def transformation_to_NN(self, x_t, translation_gpu, min_vel=list, max_vel=list):
        """
        Transform system states to normalized states in the neural network
        """

        x_t_NN = copy.deepcopy(x_t)

        #scale wrt room size:
        x_t_NN[0] = self.normalize_to_NN(x_t[0], x_min=self.min_state, x_max=self.max_state)

        #scale wrt velocities
        x_t_gpu = torch.FloatTensor(x_t_NN).cuda()
        # if self.mode_NN == "2nd": #self.mode == "acc" and
        #     x_t_gpu[0][self.dof_task:self.dof_task*2] = normalize_state(x_t_gpu[0][self.dof_task:self.dof_task*2] * self.dt, min_vel, max_vel)

        # translation wrt goal
        x_t_gpu[0][0:self.dof_task] = self.translation_to_NN(x_t_gpu[0][0:self.dof_task], translation=translation_gpu)
        x_t_cpu = x_t_gpu.T.cpu().detach().numpy()
        return x_t_gpu, x_t_cpu

    def system_quat_to_NN(self, quat, offset):
        offset_inverse = UnitQuaternion(offset).inv()
        if torch.is_tensor(quat):
            quat2 = quat.cpu().numpy()
            # quat_NN = UnitQuaternion(quat2) / UnitQuaternion(offset)
            quat_NN = self.quat_product(quat2, offset_inverse.A) #UnitQuaternion(quat2) * offset_inverse
        else:
            quat_np = quat.transpose()[0]
            quat2 = UnitQuaternion(s=quat_np[0], v=quat_np[1:], norm=True)
            if self.quaternion_flipped(quat=quat2.A, quat_prev=quat.transpose()[0]): #todo, add!!
                # print("flipped!!")
                quat2 = quat2 * -1
            # quat_NN = quat2 * offset_inverse
            quat_NN = self.quat_product(quat_np, offset_inverse.A)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [quat_NN, quat2, offset])
        return torch.FloatTensor(quat_NN).cuda() #torch.FloatTensor(quat_NN.A).cuda()