# 経験させるエピソード数
n_epochs =300

# 画面サイズ
size = 6

# ボールの落ちる間隔
interval = 4

# 棒の長さ
p_len = 2

# 環境、エージェントの初期化
env = CatchBall(size=size,interval=interval,p_len=p_len)
agent = DQNAgent(env.enable_actions,size=size)

win = 0  # キャッチした回数
total_frame = 0  # 総フレーム数
e = 0  # エピソード数

while e < n_epochs:
    # 初期化
    frame = 0  # このエピソードが始まってからのフレーム数
    loss = 0.0  # 損失関数の合計(後で平均をログとして表示する)
    Q_max = 0.0  # 各時刻での行動価値関数の最大値の合計(後で平均をログとして表示する)
    env.reset()
    state_t_1, reward_t, terminal = env.observe()  # 初期状態の観測
    win = 0
    while not terminal:  # エピソードが終わるまで実行
        state_t = state_t_1

        # 行動を選択
        action_t = agent.select_action(state_t, agent.exploration)
        
        # 行動を実行
        env.execute_action(action_t)

        # 環境を観測
        state_t_1, reward_t, terminal = env.observe()

        # 経験を保存
        start_replay = agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

        # experience replay
        if start_replay:  # メモリが貯まったら
            agent.update_exploration(e)  # εを更新
            agent.experience_replay(e)

        # 200ステップ毎にtarget networkを同期
        if total_frame % 200 == 0 and start_replay:
            agent.update_target_model()

        # ログ用
        frame += 1
        total_frame += 1
        loss += agent.current_loss
        Q_max += np.max(agent.Q_values(state_t))
        if reward_t == 1:
            win += 1

    if start_replay:        
        # ログを表示
        # 実行する時はコメントアウトを外してください。
        # print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(e+1, n_epochs, win, loss / frame, Q_max / frame))
        win = 0

    # experience replayが始まった時点からエピソードをカウント
    if start_replay:
        e += 1