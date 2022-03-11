from ipaddress import summarize_address_range
import os
from tqdm import tqdm
import numpy as np
import time

# import envs.env_v2 as env
import envs.fixed_env_real_bw_v2 as env_oracle_v2
# import envs.env as env
import envs.fixed_env as env_test
import envs.fixed_env_real_bw as env_oracle
from envs import load_trace

VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
TOTAL_CHUNK_NUM = 149
REBUF_PENALTY_LOG = 8
SMOOTH_PENALTY = 0.5

S_INFO = 16 # 
S_LEN = 2 # maximum length of states 
C_LEN = 10 # content length 

M_IN_K = 1000

TRAIN_TRACES = './envs/traces/pre_webget_1608/cooked_traces/'
SUMMARY_DIR = './Results/sim'

def main():
    start = time.process_time()
    # # test(TEST_MODEL, TEST_TRACES, LOG_FILE)
    ## load the training traces
    # all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

        ## -----------------------------------configures-------------------------------------------------
    video = 'Avengers'
    video_size_file = './envs/video_size/' + video + '/video_size_'
    video_psnr_file = './envs/video_psnr/' + video + '/chunk_psnr'

    Train_traces = TRAIN_TRACES
    log_path_ini = SUMMARY_DIR

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
    # train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
    # train_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)

    train_env_1 = env_oracle_v2.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
    train_env_1.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)

    bit_rate = int(2)
    last_quality = 0
    time_stamp = 0.
    rebuff_p = REBUF_PENALTY_LOG
    smooth_p = SMOOTH_PENALTY
    mpc_horizon = 3

    train_env = train_env_1


    # model.load_state_dict(model.state_dict())
    all_file_name = all_file_names
    log_path = log_path_ini + '_' + all_file_name[train_env.trace_idx]
    log_file = open(log_path, 'w')
    time_stamp = 0
    for video_count in tqdm(range(len(all_file_name))):
        while True:
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                    next_video_chunk_psnrs, end_of_video, video_chunk_remain, \
                        curr_chunk_sizes, curr_chunk_psnrs = train_env.get_video_chunk(bit_rate)
            
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smooth penalty
            curr_quality = curr_chunk_psnrs[bit_rate]
            reward =  curr_quality \
                        - rebuff_p * rebuf \
                            - smooth_p * np.abs(curr_quality - last_quality)
            last_quality = curr_quality

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
            log_file.flush()

            # future chunks length (try 4 if that many remaining)
            last_index = int(TOTAL_CHUNK_NUM - video_chunk_remain -1)
            future_chunk_length = mpc_horizon
            if (TOTAL_CHUNK_NUM - last_index < mpc_horizon ):
                future_chunk_length = TOTAL_CHUNK_NUM - last_index

            opt_a = train_env.solving_opt(buffer_size, bit_rate, future_chunk_length, rebuff_p, smooth_p, 6)

            bit_rate = opt_a
            
            if end_of_video:
                last_quality = 0

                log_file.write('\n')
                log_file.close()
                time_stamp = 0

                if video_count + 1 >= len(all_file_name):
                    break
                else:
                    log_path = log_path_ini + '_' + all_file_name[train_env.trace_idx]
                    log_file = open(log_path, 'w')
                    break

    end = time.process_time()
    print('finish all in %s' % str(end - start))

if __name__ == '__main__':
    main()