from ModelTrain.module.model_module import Imitate_Model
import cv2

if __name__ == '__main__':
    # model = Imitate_Model(ckpt_dir='./ckpt/ckpt_move_cube_new',ckpt_name='policy_last.ckpt')
    model = Imitate_Model(ckpt_name='policy_last.ckpt')
    model.loadModel()
    observation = {'qpos':[],'images':{'left_wrist':[],'right_wrist':[],'top':[]}}
    i=0
    while i<10:
        observation['qpos'] = [-1.57, 0, -1.57, 0, 1.57, 1.57, 1, 1.57, 0, 1.57, 0, -1.57, -1.57, 1]  #  input joint value (unit radians) and Grippers value(0~1).The 7th and 14th values are the left and right hand gripper values, respectively
        observation['images']['left_wrist'] = cv2.imread("./testimg/left_wrist.jpg", 1)  # input image
        observation['images']['right_wrist'] = cv2.imread("./testimg/right_wrist.jpg", 1)
        observation['images']['top'] = cv2.imread("./testimg/top.jpg", 1)
        cv2.imshow("img", observation['images']['left_wrist'])
        cv2.waitKey(10)
        action = model.predict(observation,i)  # out put
        print(action)
        i +=1
