'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity, matrix, zeros
from numpy.linalg import pinv, norm
from scipy.spatial.transform import Rotation


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        # YOUR CODE HERE
        length = len(self.chains[effector_name])
        J = zeros((length,6))
        
        last_joint = self.chains[effector_name][-1]
        pos_e = self.transforms[last_joint][3, : -1] - transform[3, : -1]
        rot_cur = zeros((3,3))
        rot_tar = zeros((3,3))
        for i in range(3):
            for j in range(3):
                rot_cur[i,j] = self.transforms[last_joint][i,j]
                rot_tar[i,j] = transform[i,j]
        rot_e = Rotation.from_matrix(rot_cur@rot_tar.T).as_rotvec()

        e = zeros((1,6))
        for i in range(3):
            e[0,i] = pos_e[0,i]
            e[0,i+3] = rot_e[i]
        
        x_e = transform[3, 0]
        y_e = transform[3, 1]
        z_e = transform[3, 2]
        
        for i in range(length):
            joint = self.chains[effector_name][i]
            x_i = self.transforms[joint][2, 0]
            y_i = self.transforms[joint][2, 1]
            z_i = self.transforms[joint][2, 2]
            r = self.transforms[joint][3, : 3]
            J[0, i] = (y_i-y_e)-(z_i-z_e)
            J[1, i] = (z_i-z_e)-(x_i-x_e)
            J[2, i] = (x_i-x_e)-(y_i-y_e)
            J[3, i] = r[0,0]
            J[4, i] = r[0,1]
            J[5, i] = r[0,2]

        J_inv = pinv(J)

        angles = J_inv @ e[0].T

        for i in range(len(angles)):
            joint_angles.append(angles[i, 0])

        reached_pos = False
        if norm(e[0]) < 1e-4:
            reached_pos = True

        return joint_angles, reached_pos

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        #self.keyframes = ([], [], [])  # the result joint angles have to fill in
        joint_angles_over_time = []
        for i in range(len(self.chains[effector_name])):
            joint_angles_over_time.append([])

        for i in range(1000):
            joint_angles, reached_pos = self.inverse_kinematics(effector_name, transform)
            for j in range(len(joint_angles)):
                joint_angles_over_time[j].append(joint_angles[j])
            if reached_pos:
                break
        time = []
        for i in range(0, len(joint_angles_over_time)):
            time.append(i/100)
        times = []
        for i in range(len(self.chains[effector_name])):
            times.append(time)

        self.keyframes = (self.chains[effector_name],
                            times,
                            joint_angles_over_time)

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
