import gym
import matplotlib.pyplot as plt
import numpy as np

# 変更点
PATH = 'CartPole\CartPole2\output1.npy'

# ここまで


x = []
y = []

def get_status(_observation,env):
    env_low = env.observation_space.low # 位置と速度の最小値
    env_high = env.observation_space.high #　位置と速度の最大値
    env_dx=[0]*4
    for j in range(4):
        env_dx[j] = (int(env_high[j]*(10**8))/(10**8) - int(env_low[j]*(10**8))/(10**8)) / 100 # 100等分
    
    # 0〜99の離散値に変換する
    cart_position = int((_observation[0] - env_low[0])/env_dx[0])
    cart_velocity = int((_observation[1] - env_low[1])/env_dx[1])
    pole_angle = int((_observation[2] - env_low[2])/env_dx[2])
    pole_velocity_at_tip = int((_observation[3] - env_low[3])/env_dx[3])
    
    return cart_position, cart_velocity,pole_angle,pole_velocity_at_tip


def update_q_table(_q_table, _action,  _observation, _next_observation, _reward, _episode,_env):
    
    alpha = 0.2 # 学習率
    gamma = 0.99  # 時間割引き率

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_cart_position, next_cart_velocity, next_pole_angle, next_velocity_at_tip = get_status(_next_observation, _env)
    next_max_q_value = max(_q_table[next_cart_velocity][next_pole_angle][next_velocity_at_tip])

    # 行動前の状態の行動価値 Q(s,a)
    cart_position, cart_velocity, pole_angle, pole_velocity_at_tip = get_status(_observation,_env)
    q_value = _q_table[cart_velocity][pole_angle][pole_velocity_at_tip][_action]

    # 行動価値関数の更新
    _q_table[cart_velocity][pole_angle][pole_velocity_at_tip][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value -((cart_position-next_cart_position)^2)*0.01)

    return _q_table

def get_action(_env, _q_table, _observation, _episode):
    epsilon = 0.01
    if np.random.uniform(0, 1) > epsilon:
        cart_position, cart_velocity, pole_angle, pole_velocity_at_tip = get_status(_observation,_env)
        _action = np.argmax(_q_table[cart_velocity][pole_angle][pole_velocity_at_tip])
    else:
        _action = np.random.choice([0, 1])
    # cart_position, cart_velocity, pole_angle, pole_velocity_at_tip = get_status(_observation,_env)
    # _action = np.argmax(_q_table[cart_velocity][pole_angle][pole_velocity_at_tip])
    return _action


def main():
    env = gym.make('CartPole-v0')  # make your environment!

    # Qテーブルの初期化
    q_table = np.zeros((100, 100, 100, 2))
    num=0
    a=[]
    try:
        a = np.load(PATH)  # 読み込み
        q_table = a
    except :
        None


    observation = env.reset()
    rewards = []

    # 10000エピソードで学習する
    for episode in range(50000):

        total_reward = 0
        observation = env.reset()

        for i in range(200):
            env.render()

            # ε-グリーディ法で行動を選択
            action = get_action(env, q_table, observation, episode)

            # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
            next_observation, reward, done, _ = env.step(action)

            # Qテーブルの更新
            q_table = update_q_table(q_table, action, observation, next_observation, reward, episode,env)
            total_reward += reward

            observation = next_observation
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                print('episode: {}, total_reward: {}'.format(episode, total_reward))
                rewards.append(total_reward)
                y.append(i+1)
                x.append(int(episode))
                
                break

            # if done:
            #     if episode%100 == 0:
            #         print("Episode finished after {} timesteps".format(i+1))
            #         print('episode: {}, total_reward: {}'.format(episode, total_reward))
            #         break
    np.save(PATH, q_table)
    plt.plot(x, y)        
    plt.show()
    
if __name__ =='__main__':
    main()
