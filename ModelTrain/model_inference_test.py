from ModelTrain.module.model_module import Imitate_Model
import cv2

if __name__ == '__main__':
    # model = Imitate_Model(ckpt_dir='./ckpt/ckpt_move_cube_new',ckpt_name='policy_last.ckpt')
    model = Imitate_Model(ckpt_name='policy_last.ckpt')
    model.loadModel()
    observation = {'qpos':[],'images':{'left_wrist':[],'right_wrist':[],'top':[]}}
    i=0
    while i<10:
        observation['qpos'] = [0., -0.96, 1.16, 0., -0.3, 0., 0.09984833, 0., -0.96, 1.16, 0., -0.3, 0.,
                               0.09984833]  # 注意调试获取pos_numpy时，显示的qpos_numpy正负号可能不对，因此需要用qpos_numpy[i]来显示
        observation['images']['left_wrist'] = cv2.imread("./testimg/left_wrist.jpg", 1)
        observation['images']['right_wrist'] = cv2.imread("./testimg/right_wrist.jpg", 1)
        observation['images']['top'] = cv2.imread("./testimg/top.jpg", 1)
        cv2.imshow("11", observation['images']['left_wrist'])
        cv2.waitKey(10)
        action = model.predict(observation,i)
        print(action)
        i +=1
