# -*- coding: utf-8 -*-
import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
BUFFER_THRESH = 60000.0 # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95

# pythran export get_curr_chunk_quality(int list, int, int, int, float list list)
def get_curr_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_]

# pythran export get_last_chunk_quality(int list, int, int, int, float list list)
def get_last_chunk_quality(ori_infos_int, trace_idx, video_chunk_counter, quality, chunk_psnr):
    if trace_idx == -1:
        video_chunk_counter_ = ori_infos_int[2]
    video_chunk_counter_ = video_chunk_counter
    return chunk_psnr[quality][video_chunk_counter_ - 1]

# pythran export get_download_time_upward(int list, float list, float list list, float list list, int list list, int, int, int, float, int)
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

# pythran export get_download_time_downward(int list, float list, float list list, float list list, int list list, int, int, int, float, int)
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




