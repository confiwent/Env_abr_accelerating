# pythran export get_curr_chunk_quality(int list, int, int, int, int list dict)
def get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_]

# pythran export get_last_chunk_quality(int list, int, int, int, int list dict)
def get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_ - 1]



# pythran export searching_upward(int list, float list, float [:,:], float [:,:], int:int[] dict, int:float[] dict, float, int, int, float, float, int)
def searching_upward(ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, chunk_psnr, start_buffer, last_bit_rate, future_chunk_length, rebuf_p, smooth_p, a_dim):
    max_reward = -10000000000
    reward_comparison = False
    send_data = 0
    parents_pool = [[0.0,-1,-1,-1,-1, start_buffer, int(last_bit_rate)]]
    for position in range(future_chunk_length):
        if position == future_chunk_length-1:
            reward_comparison = True
        children_pool = []
        for parent in parents_pool:
            action = 0
            trace_idx = parent[1]
            video_chunk_counter = parent[2]
            mahimahi_ptr = parent[3]
            last_mahimahi_time = parent[4]
            curr_buffer = parent[5]
            last_quality = parent[-1]
            curr_rebuffer_time = 0
            chunk_quality = action
            # get download time with true bandwidth
            download_time, download_time_upward, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ = get_download_time_upward(
                ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality)

            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += 4

            # reward
            curr_chunk_psnr = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, chunk_quality, chunk_psnr)
            last_chunk_psnr = get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, last_quality, chunk_psnr)
            bitrate_sum = curr_chunk_psnr
            smoothness_diffs = abs(curr_chunk_psnr - last_chunk_psnr)
            rebuf_penalty = rebuf_p
            reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                        smooth_p * smoothness_diffs)
            reward += parent[0]

            children = parent[:]
            children[0] = reward
            children[1] = trace_idx_
            children[2] = video_chunk_counter_
            children[3] = mahimahi_ptr_
            children[4] = last_mahimahi_time_
            children[5] = curr_buffer
            children.append(action)
            children_pool.append(children)
            if (reward >= max_reward) and reward_comparison:
                if send_data > children[7] and reward == max_reward:
                    send_data = send_data
                else:
                    send_data = children[7]  # index must be 7, not 6 or -1, since the length of children's list will increase with the tree being expanded
                max_reward = reward

            # criterion terms
            # theta = 
            rebuffer_term = rebuf_penalty * (
                        max(download_time_upward - parent[5], 0) - max(download_time - parent[5], 0))

            psnr_a = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action, chunk_psnr)
            psnr_a_ = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action + 1, chunk_psnr)
            if (action + 1 <= parent[-1]):
                High_Maybe_Superior = ((1.0 + 2 * smooth_p) * (psnr_a - psnr_a_) + rebuffer_term < 0.0)
            else:
                High_Maybe_Superior = (
                        (psnr_a - psnr_a_) + rebuffer_term < 0.0)

            while High_Maybe_Superior:
                trace_idx = parent[1]
                video_chunk_counter = parent[2]
                mahimahi_ptr = parent[3]
                last_mahimahi_time = parent[4]
                curr_buffer = parent[5]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action + 1

                download_time, download_time_upward, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ = get_download_time_upward(
                    ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size,trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality)
                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                curr_chunk_psnr = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, chunk_quality, chunk_psnr)
                last_chunk_psnr = get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, last_quality, chunk_psnr)
                bitrate_sum = curr_chunk_psnr
                smoothness_diffs = abs(curr_chunk_psnr - last_chunk_psnr)
                rebuf_penalty = rebuf_p
                reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                            smooth_p * smoothness_diffs)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = trace_idx_
                children[2] = video_chunk_counter_
                children[3] = mahimahi_ptr_
                children[4] = last_mahimahi_time_
                children[5] = curr_buffer
                children.append(chunk_quality)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[7] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[7]
                    max_reward = reward

                action += 1
                if action + 1 == a_dim:
                    break
                # criterion terms
                # theta = 
                rebuffer_term = rebuf_penalty * (
                        max(download_time_upward - parent[5], 0) - max(download_time - parent[5], 0))
                psnr_a = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action, chunk_psnr)
                psnr_a_ = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action + 1, chunk_psnr)
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * smooth_p) * (psnr_a - psnr_a_) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = (
                            (psnr_a - psnr_a_) + rebuffer_term < 0.0)

        parents_pool = children_pool

    return send_data

# pythran export searching_downward(int list, float list, float [:,:], float [:,:], int:int[] dict, int:float[] dict, float, int, int, float, float, int)
def searching_downward(ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, chunk_psnr, start_buffer, last_bit_rate, future_chunk_length, rebuf_p, smooth_p, a_dim):
    max_reward = -10000000000
    reward_comparison = False
    send_data = 0
    parents_pool = [[0.0,-1,-1,-1,-1, start_buffer, int(last_bit_rate)]]
    for position in range(future_chunk_length):
        if position == future_chunk_length-1:
            reward_comparison = True
        children_pool = []
        for parent in parents_pool:
            action = int(a_dim - 1)
            trace_idx = parent[1]
            video_chunk_counter = parent[2]
            mahimahi_ptr = parent[3]
            last_mahimahi_time = parent[4]
            curr_buffer = parent[5]
            last_quality = parent[-1]
            curr_rebuffer_time = 0
            chunk_quality = action
            # get download time with true bandwidth
            download_time, download_time_downward, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ = get_download_time_downward(
                ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality)

            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += 4

            # reward
            curr_chunk_psnr = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, chunk_quality, chunk_psnr)
            last_chunk_psnr = get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, last_quality, chunk_psnr)
            bitrate_sum = curr_chunk_psnr
            smoothness_diffs = abs(curr_chunk_psnr - last_chunk_psnr)
            rebuf_penalty = rebuf_p
            reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                        smooth_p * smoothness_diffs)
            reward += parent[0]

            children = parent[:]
            children[0] = reward
            children[1] = trace_idx_
            children[2] = video_chunk_counter_
            children[3] = mahimahi_ptr_
            children[4] = last_mahimahi_time_
            children[5] = curr_buffer
            children.append(action)
            children_pool.append(children)
            if (reward >= max_reward) and reward_comparison:
                if send_data > children[7] and reward == max_reward:
                    send_data = send_data
                else:
                    send_data = children[7]  # index must be 7, not 6 or -1, since the length of children's list will increase with the tree being expanded
                max_reward = reward

            # criterion terms
            # theta = 
            rebuffer_term = rebuf_penalty * (
                        max(download_time - parent[5], 0) - max(download_time_downward - parent[5], 0))

            action_ = action - 1
            psnr_a = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action, chunk_psnr)
            psnr_a_ = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action_, chunk_psnr)
            if (action <= parent[-1]):
                Low_is_Superior = ((1.0 + 2 * smooth_p) * (psnr_a_ - psnr_a) + rebuffer_term >= 0.0)
            else:
                Low_is_Superior = (
                        (psnr_a_ - psnr_a) + rebuffer_term >= 0.0)

            while Low_is_Superior:
                trace_idx = parent[1]
                video_chunk_counter = parent[2]
                mahimahi_ptr = parent[3]
                last_mahimahi_time = parent[4]
                curr_buffer = parent[5]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action - 1

                download_time, download_time_downward, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ = get_download_time_downward(
                    ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, 
                    trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality)
                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                curr_chunk_psnr = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, chunk_quality, chunk_psnr)
                last_chunk_psnr = get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, last_quality, chunk_psnr)
                bitrate_sum = curr_chunk_psnr
                smoothness_diffs = abs(curr_chunk_psnr - last_chunk_psnr)
                rebuf_penalty = rebuf_p
                reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                            smooth_p * smoothness_diffs)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = trace_idx_
                children[2] = video_chunk_counter_
                children[3] = mahimahi_ptr_
                children[4] = last_mahimahi_time_
                children[5] = curr_buffer
                children.append(chunk_quality)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[7] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[7]
                    max_reward = reward

                action -= 1
                if action == 0:
                    break

                rebuffer_term = rebuf_penalty * (
                        max(download_time - parent[5], 0) - max(download_time_downward - parent[5], 0))
                
                action_ = action - 1
                psnr_a = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action, chunk_psnr)
                psnr_a_ = get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, action_, chunk_psnr)
                if (action <= parent[-1]):
                    Low_is_Superior = ((1.0 + 2 * smooth_p) * (psnr_a_ - psnr_a) + rebuffer_term >= 0.0)
                else:
                    Low_is_Superior = (
                            (psnr_a_ - psnr_a) + rebuffer_term >= 0.0)

        parents_pool = children_pool
    return send_data
