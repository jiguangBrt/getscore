# 80 次语言模型实验逐次函数调用次数

口径及完整性说明见 [主报告](report.md)。3 次续跑均已合并归档前缀。返回错误的调用包含在总数中。

## 感知与观察

| 运行 | segment | back_project | view_env_state |
| --- | --- | --- | --- |
| long-original-0-gpt6-gawm | 0 | 18 | 1 |
| long-original-0-gpt6-pi05 | 0 | 24 | 2 |
| long-original-0-kimi-gawm | 3 | 7 | 1 |
| long-original-0-kimi-pi05 | 0 | 15 | 1 |
| long-original-1-gpt6-gawm | 0 | 18 | 1 |
| long-original-1-gpt6-pi05 | 0 | 21 | 1 |
| long-original-1-kimi-gawm | 0 | 18 | 1 |
| long-original-1-kimi-pi05 | 0 | 15 | 1 |
| long-original-2-gpt6-gawm | 0 | 23 | 2 |
| long-original-2-gpt6-pi05 | 0 | 26 | 1 |
| long-original-2-kimi-gawm | 4 | 44 | 1 |
| long-original-2-kimi-pi05 | 0 | 16 | 1 |
| long-original-3-gpt6-gawm | 0 | 22 | 1 |
| long-original-3-gpt6-pi05 | 0 | 27 | 2 |
| long-original-3-kimi-gawm | 3 | 23 | 1 |
| long-original-3-kimi-pi05 | 0 | 26 | 1 |
| long-original-4-gpt6-gawm | 5 | 25 | 2 |
| long-original-4-gpt6-pi05 | 11 | 6 | 2 |
| long-original-4-kimi-gawm | 1 | 26 | 1 |
| long-original-4-kimi-pi05 | 10 | 18 | 2 |
| long-original-5-gpt6-gawm | 0 | 29 | 2 |
| long-original-5-gpt6-pi05 | 0 | 23 | 1 |
| long-original-5-kimi-gawm | 1 | 9 | 1 |
| long-original-5-kimi-pi05 | 0 | 11 | 1 |
| long-original-6-gpt6-gawm | 2 | 17 | 2 |
| long-original-6-gpt6-pi05 | 3 | 19 | 1 |
| long-original-6-kimi-gawm | 9 | 19 | 2 |
| long-original-6-kimi-pi05 | 5 | 34 | 2 |
| long-original-7-gpt6-gawm | 0 | 13 | 1 |
| long-original-7-gpt6-pi05 | 0 | 23 | 1 |
| long-original-7-kimi-gawm | 0 | 20 | 1 |
| long-original-7-kimi-pi05 | 0 | 18 | 1 |
| long-original-8-gpt6-gawm | 0 | 25 | 2 |
| long-original-8-gpt6-pi05 | 0 | 45 | 1 |
| long-original-8-kimi-gawm | 3 | 17 | 1 |
| long-original-8-kimi-pi05 | 0 | 27 | 1 |
| long-original-9-gpt6-gawm | 2 | 15 | 1 |
| long-original-9-gpt6-pi05 | 3 | 22 | 1 |
| long-original-9-kimi-gawm | 6 | 14 | 1 |
| long-original-9-kimi-pi05 | 1 | 30 | 1 |
| long-swap-0-gpt6-gawm | 0 | 24 | 1 |
| long-swap-0-gpt6-pi05 | 0 | 22 | 1 |
| long-swap-0-kimi-gawm | 1 | 31 | 1 |
| long-swap-0-kimi-pi05 | 3 | 16 | 1 |
| long-swap-1-gpt6-gawm | 0 | 25 | 1 |
| long-swap-1-gpt6-pi05 | 0 | 13 | 1 |
| long-swap-1-kimi-gawm | 2 | 17 | 1 |
| long-swap-1-kimi-pi05 | 3 | 22 | 1 |
| long-swap-2-gpt6-gawm | 0 | 14 | 2 |
| long-swap-2-gpt6-pi05 | 0 | 12 | 1 |
| long-swap-2-kimi-gawm | 0 | 22 | 1 |
| long-swap-2-kimi-pi05 | 0 | 27 | 1 |
| long-swap-3-gpt6-gawm | 2 | 19 | 1 |
| long-swap-3-gpt6-pi05 | 0 | 22 | 2 |
| long-swap-3-kimi-gawm | 0 | 51 | 1 |
| long-swap-3-kimi-pi05 | 0 | 28 | 1 |
| long-swap-4-gpt6-gawm | 3 | 16 | 2 |
| long-swap-4-gpt6-pi05 | 5 | 24 | 1 |
| long-swap-4-kimi-gawm | 5 | 21 | 1 |
| long-swap-4-kimi-pi05 | 5 | 39 | 1 |
| long-swap-5-gpt6-gawm | 0 | 33 | 1 |
| long-swap-5-gpt6-pi05 | 0 | 36 | 1 |
| long-swap-5-kimi-gawm | 5 | 25 | 1 |
| long-swap-5-kimi-pi05 | 0 | 27 | 1 |
| long-swap-6-gpt6-gawm | 4 | 34 | 1 |
| long-swap-6-gpt6-pi05 | 3 | 33 | 2 |
| long-swap-6-kimi-gawm | 8 | 19 | 1 |
| long-swap-6-kimi-pi05 | 7 | 19 | 1 |
| long-swap-7-gpt6-gawm | 0 | 15 | 1 |
| long-swap-7-gpt6-pi05 | 0 | 22 | 1 |
| long-swap-7-kimi-gawm | 0 | 21 | 1 |
| long-swap-7-kimi-pi05 | 0 | 19 | 1 |
| long-swap-8-gpt6-gawm | 0 | 16 | 1 |
| long-swap-8-gpt6-pi05 | 0 | 13 | 1 |
| long-swap-8-kimi-gawm | 0 | 22 | 1 |
| long-swap-8-kimi-pi05 | 0 | 44 | 1 |
| long-swap-9-gpt6-gawm | 3 | 18 | 2 |
| long-swap-9-gpt6-pi05 | 1 | 21 | 1 |
| long-swap-9-kimi-gawm | 3 | 21 | 1 |
| long-swap-9-kimi-pi05 | 1 | 29 | 1 |

## 机器人动作

| 运行 | move_to | move_pose | pi0_pick | pi0_doubled | set_gripper | release | rotate_wrist | rotate_pitch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long-original-0-gpt6-gawm | 11 | 0 | 3 | 0 | 2 | 1 | 0 | 0 |
| long-original-0-gpt6-pi05 | 12 | 1 | 4 | 0 | 2 | 2 | 0 | 0 |
| long-original-0-kimi-gawm | 7 | 0 | 2 | 0 | 2 | 2 | 0 | 0 |
| long-original-0-kimi-pi05 | 7 | 0 | 2 | 0 | 1 | 1 | 0 | 0 |
| long-original-1-gpt6-gawm | 12 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-1-gpt6-pi05 | 12 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-1-kimi-gawm | 8 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-1-kimi-pi05 | 9 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-2-gpt6-gawm | 2 | 0 | 4 | 2 | 1 | 1 | 0 | 0 |
| long-original-2-gpt6-pi05 | 5 | 0 | 4 | 0 | 1 | 0 | 0 | 0 |
| long-original-2-kimi-gawm | 16 | 12 | 7 | 3 | 12 | 0 | 0 | 0 |
| long-original-2-kimi-pi05 | 5 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-3-gpt6-gawm | 7 | 4 | 2 | 3 | 1 | 1 | 0 | 0 |
| long-original-3-gpt6-pi05 | 8 | 0 | 1 | 4 | 0 | 1 | 0 | 0 |
| long-original-3-kimi-gawm | 8 | 7 | 1 | 3 | 1 | 2 | 0 | 0 |
| long-original-3-kimi-pi05 | 4 | 14 | 1 | 3 | 3 | 1 | 0 | 1 |
| long-original-4-gpt6-gawm | 14 | 0 | 4 | 0 | 2 | 1 | 0 | 0 |
| long-original-4-gpt6-pi05 | 12 | 0 | 2 | 0 | 2 | 1 | 1 | 0 |
| long-original-4-kimi-gawm | 13 | 0 | 3 | 0 | 3 | 1 | 4 | 0 |
| long-original-4-kimi-pi05 | 16 | 1 | 7 | 0 | 2 | 2 | 0 | 0 |
| long-original-5-gpt6-gawm | 8 | 5 | 3 | 0 | 1 | 1 | 0 | 0 |
| long-original-5-gpt6-pi05 | 10 | 1 | 6 | 0 | 1 | 1 | 0 | 0 |
| long-original-5-kimi-gawm | 7 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| long-original-5-kimi-pi05 | 5 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| long-original-6-gpt6-gawm | 9 | 2 | 8 | 0 | 0 | 1 | 0 | 0 |
| long-original-6-gpt6-pi05 | 12 | 0 | 6 | 0 | 2 | 3 | 0 | 0 |
| long-original-6-kimi-gawm | 27 | 7 | 4 | 0 | 4 | 1 | 2 | 1 |
| long-original-6-kimi-pi05 | 33 | 0 | 5 | 0 | 8 | 4 | 0 | 0 |
| long-original-7-gpt6-gawm | 10 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-7-gpt6-pi05 | 14 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-7-kimi-gawm | 8 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-7-kimi-pi05 | 11 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-original-8-gpt6-gawm | 6 | 0 | 7 | 0 | 1 | 0 | 0 | 0 |
| long-original-8-gpt6-pi05 | 23 | 0 | 4 | 0 | 5 | 2 | 2 | 0 |
| long-original-8-kimi-gawm | 16 | 0 | 8 | 0 | 4 | 0 | 0 | 0 |
| long-original-8-kimi-pi05 | 16 | 0 | 5 | 0 | 2 | 2 | 0 | 0 |
| long-original-9-gpt6-gawm | 6 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| long-original-9-gpt6-pi05 | 5 | 7 | 1 | 0 | 1 | 1 | 0 | 0 |
| long-original-9-kimi-gawm | 2 | 1 | 2 | 2 | 1 | 0 | 0 | 0 |
| long-original-9-kimi-pi05 | 18 | 8 | 8 | 1 | 6 | 2 | 0 | 0 |
| long-swap-0-gpt6-gawm | 18 | 0 | 3 | 0 | 2 | 1 | 0 | 0 |
| long-swap-0-gpt6-pi05 | 15 | 1 | 4 | 0 | 0 | 1 | 0 | 0 |
| long-swap-0-kimi-gawm | 33 | 1 | 4 | 0 | 7 | 1 | 1 | 1 |
| long-swap-0-kimi-pi05 | 13 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-1-gpt6-gawm | 14 | 0 | 3 | 0 | 2 | 2 | 0 | 0 |
| long-swap-1-gpt6-pi05 | 14 | 0 | 2 | 0 | 1 | 2 | 0 | 0 |
| long-swap-1-kimi-gawm | 13 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-1-kimi-pi05 | 10 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-2-gpt6-gawm | 2 | 0 | 3 | 1 | 0 | 0 | 0 | 0 |
| long-swap-2-gpt6-pi05 | 7 | 0 | 10 | 2 | 0 | 0 | 0 | 0 |
| long-swap-2-kimi-gawm | 8 | 1 | 3 | 0 | 2 | 0 | 0 | 0 |
| long-swap-2-kimi-pi05 | 10 | 0 | 3 | 0 | 1 | 1 | 0 | 0 |
| long-swap-3-gpt6-gawm | 10 | 2 | 3 | 2 | 0 | 1 | 0 | 0 |
| long-swap-3-gpt6-pi05 | 2 | 6 | 2 | 2 | 1 | 1 | 0 | 0 |
| long-swap-3-kimi-gawm | 21 | 11 | 3 | 1 | 1 | 3 | 0 | 0 |
| long-swap-3-kimi-pi05 | 14 | 7 | 3 | 2 | 2 | 1 | 0 | 0 |
| long-swap-4-gpt6-gawm | 13 | 0 | 3 | 0 | 2 | 1 | 1 | 0 |
| long-swap-4-gpt6-pi05 | 15 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-4-kimi-gawm | 13 | 0 | 2 | 0 | 2 | 1 | 2 | 0 |
| long-swap-4-kimi-pi05 | 22 | 9 | 4 | 0 | 3 | 3 | 0 | 0 |
| long-swap-5-gpt6-gawm | 14 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-5-gpt6-pi05 | 11 | 1 | 2 | 2 | 0 | 2 | 0 | 0 |
| long-swap-5-kimi-gawm | 10 | 1 | 1 | 0 | 1 | 0 | 1 | 0 |
| long-swap-5-kimi-pi05 | 8 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| long-swap-6-gpt6-gawm | 17 | 0 | 3 | 0 | 2 | 2 | 0 | 0 |
| long-swap-6-gpt6-pi05 | 16 | 0 | 4 | 0 | 2 | 3 | 0 | 0 |
| long-swap-6-kimi-gawm | 27 | 2 | 4 | 0 | 4 | 2 | 1 | 1 |
| long-swap-6-kimi-pi05 | 20 | 0 | 6 | 0 | 5 | 5 | 0 | 1 |
| long-swap-7-gpt6-gawm | 14 | 0 | 2 | 0 | 0 | 2 | 0 | 0 |
| long-swap-7-gpt6-pi05 | 16 | 1 | 2 | 0 | 2 | 2 | 0 | 0 |
| long-swap-7-kimi-gawm | 12 | 0 | 2 | 0 | 2 | 1 | 0 | 0 |
| long-swap-7-kimi-pi05 | 13 | 0 | 2 | 0 | 2 | 2 | 0 | 0 |
| long-swap-8-gpt6-gawm | 10 | 1 | 6 | 0 | 3 | 1 | 0 | 0 |
| long-swap-8-gpt6-pi05 | 7 | 5 | 8 | 0 | 0 | 0 | 0 | 0 |
| long-swap-8-kimi-gawm | 17 | 1 | 5 | 0 | 2 | 0 | 1 | 3 |
| long-swap-8-kimi-pi05 | 21 | 5 | 4 | 0 | 6 | 0 | 1 | 1 |
| long-swap-9-gpt6-gawm | 10 | 2 | 3 | 0 | 1 | 0 | 0 | 0 |
| long-swap-9-gpt6-pi05 | 5 | 4 | 4 | 4 | 1 | 0 | 0 | 0 |
| long-swap-9-kimi-gawm | 8 | 14 | 4 | 1 | 5 | 0 | 0 | 0 |
| long-swap-9-kimi-pi05 | 8 | 11 | 4 | 1 | 1 | 1 | 2 | 1 |

## 文件与结束操作

| 运行 | read_text_file | write_text_file | list_dir | finish |
| --- | --- | --- | --- | --- |
| long-original-0-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-original-0-gpt6-pi05 | 5 | 1 | 0 | 1 |
| long-original-0-kimi-gawm | 0 | 1 | 0 | 1 |
| long-original-0-kimi-pi05 | 7 | 1 | 1 | 1 |
| long-original-1-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-1-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-original-1-kimi-gawm | 7 | 1 | 0 | 1 |
| long-original-1-kimi-pi05 | 2 | 1 | 2 | 1 |
| long-original-2-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-2-gpt6-pi05 | 3 | 1 | 0 | 1 |
| long-original-2-kimi-gawm | 6 | 0 | 1 | 0 |
| long-original-2-kimi-pi05 | 7 | 1 | 0 | 1 |
| long-original-3-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-original-3-gpt6-pi05 | 4 | 1 | 0 | 1 |
| long-original-3-kimi-gawm | 6 | 0 | 1 | 0 |
| long-original-3-kimi-pi05 | 6 | 0 | 1 | 0 |
| long-original-4-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-4-gpt6-pi05 | 3 | 1 | 0 | 1 |
| long-original-4-kimi-gawm | 0 | 1 | 0 | 1 |
| long-original-4-kimi-pi05 | 6 | 1 | 1 | 1 |
| long-original-5-gpt6-gawm | 4 | 1 | 0 | 1 |
| long-original-5-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-original-5-kimi-gawm | 7 | 0 | 0 | 0 |
| long-original-5-kimi-pi05 | 7 | 0 | 0 | 0 |
| long-original-6-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-6-gpt6-pi05 | 3 | 1 | 0 | 1 |
| long-original-6-kimi-gawm | 6 | 1 | 1 | 1 |
| long-original-6-kimi-pi05 | 9 | 1 | 2 | 1 |
| long-original-7-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-7-gpt6-pi05 | 4 | 1 | 0 | 1 |
| long-original-7-kimi-gawm | 6 | 1 | 1 | 2 |
| long-original-7-kimi-pi05 | 6 | 1 | 1 | 1 |
| long-original-8-gpt6-gawm | 4 | 1 | 0 | 1 |
| long-original-8-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-original-8-kimi-gawm | 0 | 0 | 0 | 0 |
| long-original-8-kimi-pi05 | 6 | 0 | 1 | 0 |
| long-original-9-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-original-9-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-original-9-kimi-gawm | 6 | 1 | 1 | 1 |
| long-original-9-kimi-pi05 | 6 | 0 | 1 | 0 |
| long-swap-0-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-swap-0-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-0-kimi-gawm | 0 | 0 | 0 | 0 |
| long-swap-0-kimi-pi05 | 7 | 0 | 0 | 0 |
| long-swap-1-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-swap-1-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-1-kimi-gawm | 7 | 1 | 0 | 2 |
| long-swap-1-kimi-pi05 | 9 | 1 | 0 | 3 |
| long-swap-2-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-swap-2-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-2-kimi-gawm | 6 | 1 | 1 | 1 |
| long-swap-2-kimi-pi05 | 7 | 1 | 0 | 2 |
| long-swap-3-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-swap-3-gpt6-pi05 | 4 | 1 | 0 | 1 |
| long-swap-3-kimi-gawm | 7 | 0 | 0 | 0 |
| long-swap-3-kimi-pi05 | 6 | 0 | 1 | 0 |
| long-swap-4-gpt6-gawm | 4 | 1 | 0 | 1 |
| long-swap-4-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-4-kimi-gawm | 7 | 1 | 2 | 1 |
| long-swap-4-kimi-pi05 | 8 | 0 | 2 | 0 |
| long-swap-5-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-swap-5-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-5-kimi-gawm | 7 | 0 | 0 | 0 |
| long-swap-5-kimi-pi05 | 7 | 0 | 0 | 0 |
| long-swap-6-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-swap-6-gpt6-pi05 | 3 | 1 | 0 | 1 |
| long-swap-6-kimi-gawm | 6 | 0 | 1 | 0 |
| long-swap-6-kimi-pi05 | 7 | 1 | 0 | 1 |
| long-swap-7-gpt6-gawm | 0 | 1 | 0 | 1 |
| long-swap-7-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-7-kimi-gawm | 7 | 1 | 2 | 1 |
| long-swap-7-kimi-pi05 | 0 | 1 | 0 | 1 |
| long-swap-8-gpt6-gawm | 3 | 1 | 0 | 1 |
| long-swap-8-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-8-kimi-gawm | 7 | 0 | 0 | 0 |
| long-swap-8-kimi-pi05 | 8 | 0 | 1 | 0 |
| long-swap-9-gpt6-gawm | 4 | 1 | 0 | 1 |
| long-swap-9-gpt6-pi05 | 0 | 1 | 0 | 1 |
| long-swap-9-kimi-gawm | 6 | 0 | 1 | 0 |
| long-swap-9-kimi-pi05 | 7 | 0 | 0 | 0 |
