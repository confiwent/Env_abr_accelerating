# -*- coding: utf-8 -*-
import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
BUFFER_THRESH = 60000.0 # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95

# pythran export sim_download(float list, float list, int, float, float, float)
def sim_download(cooked_bw, cooked_time, mahimahi_ptr, last_mahimahi_time, video_chunk_size, video_chunk_counter_sent):
    while True:
        # test = cooked_time[0]
        value = 0.0
        throughput = cooked_bw[mahimahi_ptr] \
                        * B_IN_MB / BITS_IN_BYTE
        duration = cooked_time[mahimahi_ptr] \
                    - last_mahimahi_time

        packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

        if video_chunk_counter_sent + packet_payload > video_chunk_size:

            fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                throughput / PACKET_PAYLOAD_PORTION
            value += fractional_time
            last_mahimahi_time += fractional_time
            break

        video_chunk_counter_sent += packet_payload
        value += duration
        last_mahimahi_time = cooked_time[mahimahi_ptr]
        mahimahi_ptr += 1

        if mahimahi_ptr >= len(cooked_bw):
            mahimahi_ptr = 1
            last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
    return mahimahi_ptr, last_mahimahi_time, value

# pythran export sim_sleep(float, float list, float list, int, float)
def sim_sleep(buffer_size, cooked_bw, cooked_time, mahimahi_ptr, last_mahimahi_time):
    sleep_time = 0
    if buffer_size > BUFFER_THRESH:
        # exceed the buffer limit
        # we need to skip some network bandwidth here
        # but do not add up the delay
        drain_buffer_time = buffer_size - BUFFER_THRESH
        sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                        DRAIN_BUFFER_SLEEP_TIME
        buffer_size -= sleep_time

        while True:
            duration = cooked_time[mahimahi_ptr] \
                        - last_mahimahi_time
            if duration > sleep_time / MILLISECONDS_IN_SECOND:
                last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                break
            sleep_time -= duration * MILLISECONDS_IN_SECOND
            last_mahimahi_time = cooked_time[mahimahi_ptr]
            mahimahi_ptr += 1

            if mahimahi_ptr >= len(cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                mahimahi_ptr = 1
                last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
    return sleep_time, buffer_size, mahimahi_ptr, last_mahimahi_time

# pythran export get_curr_chunk_quality(int list, int, int, int, float list)
def get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_]

# pythran export get_last_chunk_quality(int list, int, int, int, float list)
def get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_ - 1]


# pythran export get_download_time_upward(int list, float list, float list, float list, int list, int, int, int, float, int)
def get_download_time_upward(ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, trace_idx, video_chunk_counter, mahimahi_ptr,last_mahimahi_time, chunk_quality):
    ## ---------------- compute last time ----------------------------------------------------
    # ori_infos_int = [self.trace_idx, self.mahimahi_ptr, self.video_chunk_counter, self.total_chunk_num, self.mahimahi_start_ptr]
    # ori_infos_float = [self.last_mahimahi_time]
    total_chunk_num = ori_infos_int[3]
    mahimahi_start_ptr = ori_infos_int[4]
    if trace_idx == -1:
        trace_idx = ori_infos_int[0]
        video_chunk_counter = ori_infos_int[2]
        mahimahi_ptr = ori_infos_int[1]
        cooked_time = all_cooked_time[trace_idx]
        last_mahimahi_time = ori_infos_float[0]
    ## ----------------- assign values ----------------------------------------------------

    cooked_bw = all_cooked_bw[trace_idx]
    cooked_time = all_cooked_time[trace_idx]

    ## ------------------- compute true bandwidth --------------------------------------------
    download_time = []
    for quality in range(chunk_quality, min(chunk_quality + 2, 6)):
        duration_all = 0
        video_chunk_counter_sent = 0  # in bytes
        video_chunk_size = video_size[quality][video_chunk_counter]
        mahimahi_ptr_tmp = mahimahi_ptr
        last_mahimahi_time_tmp = last_mahimahi_time

        while True:  # download video chunk over mahimahi
            throughput = cooked_bw[mahimahi_ptr_tmp] \
                            * B_IN_MB / BITS_IN_BYTE
            duration = cooked_time[mahimahi_ptr_tmp] \
                        - last_mahimahi_time_tmp

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                    throughput / PACKET_PAYLOAD_PORTION
                last_mahimahi_time_tmp += fractional_time
                duration_all += fractional_time
                break
            video_chunk_counter_sent += packet_payload
            last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
            mahimahi_ptr_tmp += 1

            if mahimahi_ptr_tmp >= len(cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                mahimahi_ptr_tmp = 1
                last_mahimahi_time_tmp = 0
            duration_all += duration
        download_time.append(duration_all)
        if quality == chunk_quality:
            trace_idx_ = trace_idx
            video_chunk_counter_ = video_chunk_counter
            mahimahi_ptr_ = mahimahi_ptr_tmp
            last_mahimahi_time_ = last_mahimahi_time_tmp

    ## -------------------- test whether end of video ---------------------------------------------------
    video_chunk_counter_ += 1
    if video_chunk_counter_ >= total_chunk_num:

        video_chunk_counter_ = 0
        trace_idx_ += 1
        if trace_idx_ >= len(all_cooked_time):
            trace_idx_ = 0

        cooked_time = all_cooked_time[trace_idx_]
        cooked_bw = all_cooked_bw[trace_idx_]

        # randomize the start point of the video
        # note: trace file starts with time 0
        mahimahi_ptr_ = mahimahi_start_ptr
        last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


    if len(download_time)==1:
        return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
    else:
        return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_

# pythran export get_download_time_downward(int list, float list, float list, float list, int list, int, int, int, float, int)
def get_download_time_downward(ori_infos_int, ori_infos_float, all_cooked_time, all_cooked_bw, video_size, trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality):
    ## ---------------- compute last time ----------------------------------------------------
    # ori_infos_int = [self.trace_idx, self.mahimahi_ptr, self.video_chunk_counter, self.total_chunk_num, self.mahimahi_start_ptr]
    # ori_infos_float = [self.last_mahimahi_time]
    total_chunk_num = ori_infos_int[3]
    mahimahi_start_ptr = ori_infos_int[4]
    if trace_idx == -1:
        trace_idx = ori_infos_int[0]
        video_chunk_counter = ori_infos_int[2]
        mahimahi_ptr = ori_infos_int[1]
        cooked_time = all_cooked_time[trace_idx]
        last_mahimahi_time = ori_infos_float[0]
    ## ----------------- assign values ----------------------------------------------------

    cooked_bw = all_cooked_bw[trace_idx]
    cooked_time = all_cooked_time[trace_idx]

    ## ------------------- compute true bandwidth --------------------------------------------
    download_time = []
    for quality in range(chunk_quality, max(chunk_quality - 2, -1), -1):
        duration_all = 0
        video_chunk_counter_sent = 0  # in bytes
        video_chunk_size = video_size[quality][video_chunk_counter]
        mahimahi_ptr_tmp = mahimahi_ptr
        last_mahimahi_time_tmp = last_mahimahi_time

        while True:  # download video chunk over mahimahi
            throughput = cooked_bw[mahimahi_ptr_tmp] \
                            * B_IN_MB / BITS_IN_BYTE
            duration = cooked_time[mahimahi_ptr_tmp] \
                        - last_mahimahi_time_tmp

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                    throughput / PACKET_PAYLOAD_PORTION
                last_mahimahi_time_tmp += fractional_time
                duration_all += fractional_time
                break
            video_chunk_counter_sent += packet_payload
            last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
            mahimahi_ptr_tmp += 1

            if mahimahi_ptr_tmp >= len(cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                mahimahi_ptr_tmp = 1
                last_mahimahi_time_tmp = 0
            duration_all += duration
        download_time.append(duration_all)
        if quality == chunk_quality:
            trace_idx_ = trace_idx
            video_chunk_counter_ = video_chunk_counter
            mahimahi_ptr_ = mahimahi_ptr_tmp
            last_mahimahi_time_ = last_mahimahi_time_tmp

    ## -------------------- test whether end of video ---------------------------------------------------
    video_chunk_counter_ += 1
    if video_chunk_counter_ >= total_chunk_num:

        video_chunk_counter_ = 0
        trace_idx_ += 1
        if trace_idx_ >= len(all_cooked_time):
            trace_idx_ = 0

        cooked_time = all_cooked_time[trace_idx_]
        cooked_bw = all_cooked_bw[trace_idx_]

        # randomize the start point of the video
        # note: trace file starts with time 0
        mahimahi_ptr_ = mahimahi_start_ptr
        last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


    if len(download_time)==1:
        return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
    else:
        return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_

# pythran export searching_upward(int list, float list, float list, float list, int list, float list, float, int, int, float, float, int)
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

# pythran export searching_downward(int list, float list, float list, float list, int list, float list, float, int, int, float, float, int)
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