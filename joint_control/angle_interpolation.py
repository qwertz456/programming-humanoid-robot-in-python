'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        if 'LHipYawPitch' in target_joints:
            target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE
        for i in range (len(keyframes[0])):
            times = keyframes[1][i]

            t = perception.time % keyframes[1][i][-1]
            i_minus_1 = -1
            for q in range (len(times)-1):
                if times[q] < t and times[q+1] > t:
                    i_minus_1 = q

            angles = keyframes[2][i]
            for a in range(len(angles)):
                if type(angles[a]) == list:
                    angles[a] = angles[a][0]
            solution = [0, 0, 0, 0]
            m0 = 0

            if i_minus_1 == 0 or i_minus_1 == len(times)-1:
                m1 = 0
            else:
                if (angles[i_minus_1+1]-angles[i_minus_1])*(angles[i_minus_1]-angles[i_minus_1-1]) < 0:
                    m1 = 0
                else:
                    m1 = (angles[i_minus_1+1] - 2*angles[i_minus_1] + angles[i_minus_1-1])/2
            if i_minus_1 == len(times)-1 or i_minus_1 == len(times)-2:
                m0 = 0
            else:
                if (angles[i_minus_1+2]-angles[i_minus_1+1])*(angles[i_minus_1+1]-angles[i_minus_1]) < 0:
                    m0 = 0
                else:
                    m0 = (angles[i_minus_1+2] - 2*angles[i_minus_1+1] + angles[i_minus_1])/2
                
            c = [[times[i_minus_1+1]**3, times[i_minus_1+1]**2, times[i_minus_1+1], 1],
                     [times[i_minus_1]**3, times[i_minus_1]**2, times[i_minus_1], 1],
                     [3*(times[i_minus_1+1]**2), 2*times[i_minus_1+1], 1, 0],
                     [3*(times[i_minus_1]**2), 2*times[i_minus_1], 1, 0]]
            f = [angles[i_minus_1+1], angles[i_minus_1], m1, m0]
                
            #Solving equation according to script of modul "Wissenschaftliches Rechnen"
            for elem in range(4):
                #Pivoting:
                max_elem = elem
                for element in range(elem, 4):
                    if abs(c[element][elem]) > abs(c[max_elem][elem]):
                        max_elem = element
                    c[elem], c[max_elem] = c[max_elem], c[elem]
                    f[elem], f[max_elem] = f[max_elem], f[elem]
                #Elimination:
                for row in range(elem+1, 4):
                    fac = c[row][elem]/c[elem][elem]
                    for column in range(elem, 4):
                        c[row][column] -= fac*c[elem][column]
                    f[row] -= fac*f[elem]

            for k in range(3, -1, -1):
                values_sum = 0
                for l in range(k+1, 4):
                    values_sum += c[k][l]*solution[l]
                solution[k] = (f[k] - values_sum)/c[k][k]
            func_val = solution[0]*t**3+solution[1]*t**2+solution[2]*t+solution[3]
            target_joints.update({keyframes[0][i]: func_val})

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
