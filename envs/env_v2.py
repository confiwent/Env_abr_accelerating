import numpy as np
import random
import numba
from numba import jit 
from numba import njit 
from numba.typed import List

from .env_oracle_pythran import sim_download, sim_sleep

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
# BITRATE_LEVELS = 6
# TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1

# @jit(nopython = True)
# def sim_download(cooked_bw: list, cooked_time: list, mahimahi_ptr: int, last_mahimahi_time: float, video_chunk_size: float, video_chunk_counter_sent: float, value: float):
#     while True:
#         # test = cooked_time[0]
#         throughput = cooked_bw[mahimahi_ptr] \
#                         * B_IN_MB / BITS_IN_BYTE
#         duration = cooked_time[mahimahi_ptr] \
#                     - last_mahimahi_time

#         packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

#         if video_chunk_counter_sent + packet_payload > video_chunk_size:

#             fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
#                                 throughput / PACKET_PAYLOAD_PORTION
#             value += fractional_time
#             last_mahimahi_time += fractional_time
#             break

#         video_chunk_counter_sent += packet_payload
#         value += duration
#         last_mahimahi_time = cooked_time[mahimahi_ptr]
#         mahimahi_ptr += 1

#         if mahimahi_ptr >= len(cooked_bw):
#             mahimahi_ptr = 1
#             last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
#     return mahimahi_ptr, last_mahimahi_time, value

# @jit(nopython = True)
# def sim_sleep(buffer_size, cooked_bw, cooked_time, mahimahi_ptr, last_mahimahi_time):
#     sleep_time = 0
#     if buffer_size > BUFFER_THRESH:
#         # exceed the buffer limit
#         # we need to skip some network bandwidth here
#         # but do not add up the delay
#         drain_buffer_time = buffer_size - BUFFER_THRESH
#         sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
#                         DRAIN_BUFFER_SLEEP_TIME
#         buffer_size -= sleep_time

#         while True:
#             duration = cooked_time[mahimahi_ptr] \
#                         - last_mahimahi_time
#             if duration > sleep_time / MILLISECONDS_IN_SECOND:
#                 last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
#                 break
#             sleep_time -= duration * MILLISECONDS_IN_SECOND
#             last_mahimahi_time = cooked_time[mahimahi_ptr]
#             mahimahi_ptr += 1

#             if mahimahi_ptr >= len(cooked_bw):
#                 # loop back in the beginning
#                 # note: trace file starts with time 0
#                 mahimahi_ptr = 1
#                 last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
#     return sleep_time, buffer_size, mahimahi_ptr, last_mahimahi_time

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, video_size_file, video_psnr_file, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        
        self.s_info = 17
        self.s_len = 10
        self.c_len = 3
        self.bitrate_version = [300, 750, 1200, 1850, 2850, 4300]
        self.br_dim = len(self.bitrate_version)
        self.rebuff_p = 2.66
        self.smooth_p = 1

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.cooked_time_ = List()
        self.cooked_bw_ = List()

        for i in range(len(self.cooked_time)):
            self.cooked_time_.append(self.cooked_time[i])
            self.cooked_bw_.append(self.cooked_bw[i])

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.br_dim):
            self.video_size[bitrate] = []
            with open(video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.chunk_psnr = {} # video quality of chunks
        for bitrate in range(self.br_dim):
            self.chunk_psnr[bitrate] = []
            with open(video_psnr_file + str(bitrate)) as f:
                for line in f:
                    self.chunk_psnr[bitrate].append((float(line.split()[0])))

        self.total_chunk_num = len(self.video_size[0])
        self.chunk_length_max = self.total_chunk_num

    # pythran export set_env_info(int, int, int, int, int list, float, float)
    def set_env_info(self, s_info, s_len, c_len, chunk_num, br_version, rebuff_p, smooth_p):
        self.s_info = s_info
        self.s_len = s_len
        self.c_len = c_len
        self.total_chunk_num = chunk_num
        self.chunk_length_max = chunk_num
        self.bitrate_version = br_version
        self.br_dim = len(self.bitrate_version)
        self.rebuff_p = rebuff_p
        self.smooth_p = smooth_p

    # pythran export get_env_info(None)
    def get_env_info(self):
        return self.s_info, self.s_len , self.c_len, self.total_chunk_num, self.bitrate_version, self.rebuff_p, self.smooth_p

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.br_dim

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        mahimahi_ptr, last_mahimahi_time, delay = sim_download(self.cooked_bw, self.cooked_time, self.mahimahi_ptr, self.last_mahimahi_time, float(video_chunk_size), float(video_chunk_counter_sent))
        self.mahimahi_ptr = mahimahi_ptr
        self.last_mahimahi_time = last_mahimahi_time
        # for i in range(len(value)):
        #     delay += float(value[i])

        # while True:  # download video chunk over mahimahi
        #     throughput = self.cooked_bw[self.mahimahi_ptr] \
        #                  * B_IN_MB / BITS_IN_BYTE
        #     duration = self.cooked_time[self.mahimahi_ptr] \
        #                - self.last_mahimahi_time

        #     packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

        #     if video_chunk_counter_sent + packet_payload > video_chunk_size:

        #         fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
        #                           throughput / PACKET_PAYLOAD_PORTION
        #         delay += fractional_time
        #         self.last_mahimahi_time += fractional_time
        #         break

        #     video_chunk_counter_sent += packet_payload
        #     delay += duration
        #     self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
        #     self.mahimahi_ptr += 1

        #     if self.mahimahi_ptr >= len(self.cooked_bw):
        #         # loop back in the beginning
        #         # note: trace file starts with time 0
        #         self.mahimahi_ptr = 1
        #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        # sleep_time = 0
        sleep_time, buffer_size, ptr_, last_time_ = sim_sleep(self.buffer_size, self.cooked_bw, self.cooked_time, self.mahimahi_ptr, self.last_mahimahi_time)
        self.mahimahi_ptr = ptr_
        self.last_mahimahi_time = last_time_
        self.buffer_size = buffer_size

        # sleep_time = 0
        # if self.buffer_size > BUFFER_THRESH:
        #     # exceed the buffer limit
        #     # we need to skip some network bandwidth here
        #     # but do not add up the delay
        #     drain_buffer_time = self.buffer_size - BUFFER_THRESH
        #     sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
        #                  DRAIN_BUFFER_SLEEP_TIME
        #     self.buffer_size -= sleep_time

        #     while True:
        #         duration = self.cooked_time[self.mahimahi_ptr] \
        #                    - self.last_mahimahi_time
        #         if duration > sleep_time / MILLISECONDS_IN_SECOND:
        #             self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
        #             break
        #         sleep_time -= duration * MILLISECONDS_IN_SECOND
        #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
        #         self.mahimahi_ptr += 1

        #         if self.mahimahi_ptr >= len(self.cooked_bw):
        #             # loop back in the beginning
        #             # note: trace file starts with time 0
        #             self.mahimahi_ptr = 1
        #             self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        curr_chunk_sizes = []
        curr_chunk_psnrs = []
        for i in range(self.br_dim):
            curr_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            curr_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunk_num - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_chunk_num:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            # self.total_chunk_num = random.randint(10, int(self.chunk_length_max))
            
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            self.cooked_time_ = List()
            self.cooked_bw_ = List()

            for i in range(len(self.cooked_time)):
                self.cooked_time_.append(self.cooked_time[i])
                self.cooked_bw_.append(self.cooked_bw[i])

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        next_video_chunk_psnrs = []
        for i in range(self.br_dim):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            next_video_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            next_video_chunk_psnrs, \
            end_of_video, \
            video_chunk_remain, \
            curr_chunk_sizes, \
            curr_chunk_psnrs
