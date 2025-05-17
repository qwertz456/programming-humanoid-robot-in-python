'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity, eye

from numpy import cos, sin

from recognize_posture import PostureRecognitionAgent

links = {
    #Head
    'HeadYaw': [0.0, 0.0, 0.1265],
    'HeadPitch': [0.0, 0.0, 0.0],

    #LArm
    'LShoulderPitch': [0.00, 0.098, 0.100],
    'LShoulderRoll': [0.0, 0.0, 0.0],
    'LElbowYaw': [0.105, 0.015, 0.0],
    'LElbowRoll': [0.0, 0.0, 0.0],
    'LWristYaw': [0.05595, 0.0, 0.0],
            
    #LLeg
    'LHipYawPitch': [0.0, 0.050, -0.085],
    'LHipRoll': [0.0, 0.0, 0.0],
    'LHipPitch': [0.0, 0.0, 0.0],
    'LKneePitch': [0.00, 0.00, -0.1],
    'LAnklePitch': [0.00, 0.00, -0.1029],
    'LAnkleRoll': [0.0, 0.0, 0.0],

    #RArm
    'RShoulderPitch': [0.00, 0.098, 0.100],
    'RShoulderRoll': [0.0, 0.0, 0.0],
    'RElbowYaw': [0.105, 0.015, 0.0],
    'RElbowRoll': [0.0, 0.0, 0.0],
    'RWristYaw': [0.05595, 0.0, 0.0],
            
    #RLeg
    'RHipYawPitch': [0.0, 0.050, -0.085],
    'RHipRoll': [0.0, 0.0, 0.0],
    'RHipPitch': [0.0, 0.0, 0.0],
    'RKneePitch': [0.00, 0.00, -0.1],
    'RAnklePitch': [0.00, 0.00, -0.1029],
    'RAnkleRoll': [0.0, 0.0, 0.0]
}


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       # YOUR CODE HERE
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw']
                       }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = identity(4)
        # YOUR CODE HERE
        cos_angle = cos(joint_angle)
        sin_angle = sin(joint_angle)

        R_x = eye(3)
        if joint_name in ['RElbowYaw', 'LElbowYaw', 'LHipRoll', 'RHipRoll', 'LAnkleRoll', 'RAnkleRoll']:
            R_x[0] = [1, 0, 0]
            R_x[1] = [0, cos_angle, -sin_angle]
            R_x[2] = [0, sin_angle, cos_angle]

        R_y = eye(3)
        if joint_name in ['HeadPitch', 'RShoulderPitch', 'LShoulderPitch', 'LHipYawPitch', 'RHipYawPitch', 'LHipPitch', 'RHipPitch', 'LKneePitch', 'RKneePitch', 'LAnklePitch', 'RAnklePitch']:
            R_y[0] = [cos_angle, 0, sin_angle]
            R_y[1] = [0, 1, 0]
            R_y[2] = [-sin_angle, 0, cos_angle]

        R_z = eye(3)
        if joint_name in ['HeadYaw', 'RShoulderRoll', 'RElbowRoll', 'LShoulderRoll', 'LElbowRoll']:
            R_z[0] = [cos_angle, sin_angle, 0]
            R_z[1] = [-sin_angle, cos_angle, 0]
            R_z[2] = [0, 0, 1]

        R = R_x*R_y*R_z
        for i in range(3):
            for j in range(3):
                T[i, j] = R[i, j]
            T[i, 3] = 0.0
            T[3, i] = links[joint_name][i]
        T[3, 3] = 1.0

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T = Tl@T

                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
