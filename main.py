import argparse
from email.policy import default
import sys
from operator import add
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from algos.train_im import train_iml
from algos.train_ppo import train_ppo
# from algos.train_dppo import train_dppo_pure
from algos.test import test
import envs.env as env
import envs.fixed_env as env_test
import envs.fixed_env_real_bw as env_oracle
from envs import load_trace

IMITATION_TRAIN_EPOCH = 2550
HIDDEN_LAYERS = [128, 256, 64]

# Parameters of envs
S_INFO = 16 # 
S_LEN = 2 # maximum length of states 
C_LEN = 10 # content length 
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
TOTAL_CHUNK_NUM = 149
REBUF_PENALTY_LOG = 8
SMOOTH_PENALTY = 0.5

TEST_LOG_FILE_OBE = './Results/test/oboe/'
TEST_LOG_FILE_FCC = './Results/test/fcc/'
TEST_LOG_FILE_GHT = './Results/test/ghent/'
TEST_LOG_FILE_FHN = './Results/test/fh_noisy/'
TEST_MODEL_ACT = './saved_models/mppo/0308/policy_mppo_22000.model'
TEST_MODEL_VAE = './saved_models/mppo/0308/VAE_mppo_22000.model'
TEST_TRACES_FCC = './envs/traces/pre_webget_1608/test_traces/'
TEST_TRACES_OBE = './envs/traces/traces_oboe/'
TEST_TRACES_GHT = './envs/traces/test_traces_4g2/'
TEST_TRACES_FHN = './envs/traces/test_traces_noisy/'
# TRAIN_TRACES = './envs/traces/fcc_noisy/cooked_traces/'#'./envs/traces/pre_webget_1608/cooked_traces/'
# VALID_TRACES = './envs/traces/fcc_noisy/cooked_test_traces/'#'./envs/traces/pre_webget_1608/cooked_test_traces/'
TRAIN_TRACES = './envs/traces/pre_webget_1608/cooked_traces/'
VALID_TRACES = './envs/traces/pre_webget_1608/cooked_test_traces/'
SUMMARY_DIR = './Results/sim'
MODEL_DIR = './saved_models'


parser = argparse.ArgumentParser(description='IMRL-based ABR')
parser.add_argument('--test', action='store_true', help='Evaluate only')
parser.add_argument('--adapt', action='store_true', help='Adaptation to new environments')
parser.add_argument('--im', action='store_true', help='Train policy with Imitation Learning')
parser.add_argument('--imni', action='store_true', help='Train policy with Imitation Learning without mutual information')
parser.add_argument('--mppo', action='store_true', help='Train policy with Meta PPO')
parser.add_argument('--ppo', action='store_true', help='Train policy with pure PPO')
parser.add_argument('--imrl', action='store_true', help='Train policy with pure imitation learning and meta rl')
# parser.add_argument('--dppo', action='store_true', help='Train policy with pure DPPO')
parser.add_argument('--geser', action='store_true', help='Train policy with pure GeSER MM21')
parser.add_argument('--name', default='mppo', help='the name of result folder')
parser.add_argument('--prior-not', action='store_false', help='Not update the prior during the training')
parser.add_argument('--up-vae', action='store_false', help='Not update the vae during the second phase')
parser.add_argument('--valid-i',nargs='?', const=1000, default=1000, type=int, help='The valid interval')
parser.add_argument('--prior-ui',nargs='?', const=500, default=500, type=int, help='The update interval of the prior')
parser.add_argument('--latent-dim', nargs='?', const=15, default=15, type=int, help='The dimension of latent space')
parser.add_argument('--belief', action='store_true', help='User the belief representation as one of the policy inputs')
parser.add_argument('--mpc-h', nargs='?', const=7, default=7, type=int, help='The MPC planning horizon')
parser.add_argument('--kld-beta', nargs='?', const=0.1, default=0.1, type=float, help='The coefficient of kld in the VAE loss function')
parser.add_argument('--kld-lambda', nargs='?', const=0.25, default=0.25, type=float, help='The coefficient of kld in the VAE recon loss function') ## control the strength of over-fitting of reconstruction, KL divergence between the prior P(D) and the distribution of P(D|\theta)
parser.add_argument('--vae-gamma', nargs='?', const=0.7, default=0.7, type=float, help='The coefficient of reconstruction loss in the VAE loss function')
parser.add_argument('--lc-alpha', nargs='?', const=1, default=1, type=float, help='The coefficient of cross entropy in the actor loss function')
parser.add_argument('--lc-beta', nargs='?', const=0.8, default=0.8, type=float, help='The coefficient of entropy in the imitation loss function')
parser.add_argument('--lc-beta-p', nargs='?', const=0.2, default=0.2, type=float, help='The coefficient of entropy in the actor loss function of PPO')
parser.add_argument('--lc-gamma', nargs='?', const=0.4, default=0.4, type=float, help='The coefficient of mutual information in the actor loss function')
parser.add_argument('--sp-n', nargs='?', const=10, default=10, type=int, help='The sample numbers of the mutual information')
parser.add_argument('--vpre-num', nargs='?', const=800, default=800, type=int, help='The update epoch for pretrain critic network')
parser.add_argument('--gae-gamma', nargs='?', const=0.99, default=0.99, type=float, help='The gamma coefficent for GAE estimation')
parser.add_argument('--gae-lambda', nargs='?', const=0.95, default=0.95, type=float, help='The lambda coefficent for GAE estimation')
parser.add_argument('--batch-size', nargs='?', const=64, default=64, type=int, help='Minibatch size for training')
parser.add_argument('--ppo-ups', nargs='?', const=4, default=4, type=int, help='Update numbers in each epoch for PPO')
parser.add_argument('--explo-num', nargs='?', const=2, default=2, type=int, help='Exploration steps for roll-out')
parser.add_argument('--ro-len', nargs='?', const=128, default=128, type=int, help='Length of roll-out')
parser.add_argument('--clip', nargs='?', const=0.04, default=0.04, type=float, help='Clip value of ppo')
parser.add_argument('--anneal-p', nargs='?', const=0.95, default=0.95, type=float, help='Annealing parameters for entropy regularization')
parser.add_argument('--Avengers', action='store_true', help='Use the video of Avengers')
parser.add_argument('--LasVegas', action='store_true', help='Use the video of LasVegas')
parser.add_argument('--Dubai', action='store_true', help='Use the video of Dubai')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--tg', action='store_true', help='Use Ghent traces')
parser.add_argument('--tn', action='store_true', help='Use FH-Noisy traces')

# USE_CUDA = torch.cuda.is_available()
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def main():
    # # test(TEST_MODEL, TEST_TRACES, LOG_FILE)
    ## load the training traces
    # all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    args = parser.parse_args()
    if args.test:
        ## -----------------------------------configures-------------------------------------------------

        add_str = args.name

        if args.tf:
            log_path = TEST_LOG_FILE_FCC + 'log_test_' + add_str
            test_traces = TEST_TRACES_FCC
        elif args.to:
            log_path = TEST_LOG_FILE_OBE + 'log_test_' + add_str
            test_traces = TEST_TRACES_OBE
        elif args.tg:
            log_path = TEST_LOG_FILE_GHT + 'log_test_' + add_str
            test_traces = TEST_TRACES_GHT
        elif args.tn:
            log_path = TEST_LOG_FILE_FHN + 'log_test_' + add_str
            test_traces = TEST_TRACES_FHN
        else:
            print("Please choose the throughput data traces!!!")
        
        test_model_ = [TEST_MODEL_ACT, TEST_MODEL_VAE]

        if args.Avengers:
            video = 'Avengers'
        elif args.LasVegas:
            video = 'LasVegas'
        elif args.Dubai:
            video = 'Dubai'
        video_size_file = './envs/video_size/' + video + '/video_size_'
        video_psnr_file = './envs/video_psnr/' + video + '/chunk_psnr'

        # -----------------------------initialize the environment----------------------------------------
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
        test_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw, all_file_names = all_file_names,
                                    video_size_file = video_size_file, video_psnr_file=video_psnr_file)

        s_len = 6 if args.ppo or args.geser else S_LEN
        test_env.set_env_info(S_INFO, s_len, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        
        # --------------------------------------execute--------------------------------------------------
        test(args, test_model_, test_env, HIDDEN_LAYERS, log_path)
    else:
        if torch.cuda.is_available():
                torch.cuda.set_device(0) # ID of GPU to be used
                print("CUDA Device: %d" %torch.cuda.current_device())
        ## -----------------------------------configures-------------------------------------------------
        if args.Avengers:
            video = 'Avengers'
        elif args.LasVegas:
            video = 'LasVegas'
        elif args.Dubai:
            video = 'Dubai'
        video_size_file = './envs/video_size/' + video + '/video_size_'
        video_psnr_file = './envs/video_psnr/' + video + '/chunk_psnr'

        Train_traces = TRAIN_TRACES
        Valid_traces = VALID_TRACES
        log_dir_path = SUMMARY_DIR

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Valid_traces)
        valid_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, all_file_names = all_file_names,
                                video_size_file = video_size_file, video_psnr_file=video_psnr_file)

        s_len = 6 if args.ppo or args.geser else S_LEN
        valid_env.set_env_info(S_INFO, s_len, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)

        if args.im:
            # -----------------------------initialize the environment----------------------------------------
            add_str = args.name 

            # =========== Load the model parameters =======================
            # model_vae_para = None 
            if args.adapt:
                model_vae_para = torch.load('./saved_models/im/VAE_iml_1800.model')
                model_actor_para = torch.load('./saved_models/im/policy_iml_1800.model')
            else:
                model_vae_para = None
                model_actor_para = None


            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
            train_env_1 = env_oracle.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
            train_env_1.set_env_info(S_INFO, s_len, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
            
            # ------------------------------------First stage---------------------------------------
            # Imitation meta learning
            model_actor_para, model_vae_para = train_iml(IMITATION_TRAIN_EPOCH, model_actor_para, model_vae_para, HIDDEN_LAYERS, train_env_1, valid_env, args, add_str, log_dir_path)

            # # save models in the First stage
            model_save_dir = MODEL_DIR + '/' + add_str
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            # command = 'rm ' + SUMMARY_DIR + add_str + '/*'5
            # os.system(command)
            model_actor_save_path = model_save_dir + "/%s_%s_%d.model" %(str('Policy'), add_str, int(IMITATION_TRAIN_EPOCH))
            model_vae_save_path = model_save_dir + "/%s_%s_%d.model" %(str('VAE'), add_str, int(IMITATION_TRAIN_EPOCH))
            if os.path.exists(model_actor_save_path): os.system('rm ' + model_actor_save_path)
            if os.path.exists(model_vae_save_path): os.system('rm ' + model_vae_save_path)
            torch.save(model_actor_para, model_actor_save_path)
            torch.save(model_vae_para, model_vae_save_path)

            ## -----------------------Second stage--------------------------------------
            # Meta reinforcement learning 

            # # load the models
            # model_actor_para = torch.load('./models/init/actor_bmpc_3050.model')
            # model_critic_para = torch.load('./models/init/critic_bmpc_3050.model')

            ## fine-tune the models with PPO
            # train_ppo_ft(model_actor_para, model_critic_para)
            # train_ppo_ft(model_actor_para, model_critic_para, train_env, args, qoe_metric, add_str, IMITATION_TRAIN_EPOCH)
        elif args.mppo:
            ## load the models
            # model_vae_para = None 
            if args.adapt:
                model_vae_para = torch.load('./saved_models/im/VAE_iml_1800.model')
                model_actor_para = torch.load('./saved_models/im/policy_iml_1800.model')
                model_critic_para = torch.load()
            else:
                model_vae_para = None
                model_actor_para = None
                model_critic_para = None
            # add_str = 'mppo'
            add_str = args.name 
            ## setting the envs
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
            train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
            train_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
            train_ppo(HIDDEN_LAYERS, model_vae_para, model_actor_para, model_critic_para, train_env, valid_env, args, add_str, log_dir_path)

        # elif args.imrl: # training with a meta-learning approach, including imitation learning-based initialization and reinforcement learning-based improvement
        #     add_str = args.name

        #     ## imitation part
        #     # all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        #     # train_env_1 = env_oracle.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        #     # train_env_1.set_env_info(S_INFO, s_len, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
            
        #     # # ------------------------------------First stage---------------------------------------
        #     # # Imitation meta learning
        #     # model_actor_para, model_critic_para, model_vae_para = train_iml_v2(IMITATION_TRAIN_EPOCH, HIDDEN_LAYERS, train_env_1, valid_env, args, add_str, log_dir_path)

        #     # # # save models in the First stage
        #     # model_save_dir = MODEL_DIR + '/' + add_str
        #     # if not os.path.exists(model_save_dir):
        #     #     os.mkdir(model_save_dir)
        #     # # command = 'rm ' + SUMMARY_DIR + add_str + '/*'5
        #     # # os.system(command)
        #     # model_actor_save_path = model_save_dir + "/%s_%s_%d.model" %(str('Policy'), add_str, int(IMITATION_TRAIN_EPOCH))
        #     # model_critic_save_path = model_save_dir + "/%s_%s_%d.model" %(str('Critic'), add_str, int(IMITATION_TRAIN_EPOCH))
        #     # model_vae_save_path = model_save_dir + "/%s_%s_%d.model" %(str('VAE'), add_str, int(IMITATION_TRAIN_EPOCH))
        #     # if os.path.exists(model_actor_save_path): os.system('rm ' + model_actor_save_path)
        #     # if os.path.exists(model_critic_save_path): os.system('rm ' + model_critic_save_path)
        #     # if os.path.exists(model_vae_save_path): os.system('rm ' + model_vae_save_path)
        #     # torch.save(model_actor_para, model_actor_save_path)
        #     # torch.save(model_critic_para, model_critic_save_path)
        #     # torch.save(model_vae_para, model_vae_save_path)

        #     # DEBUG    
        #     model_vae_save_path = './saved_models/imrl/VAE_imrl_550.model'
        #     model_actor_save_path = './saved_models/imrl/Policy_imrl_550.model'
        #     model_critic_save_path = './saved_models/imrl/Critic_imrl_550.model'

        #     # RL part
        #     model_vae_para = torch.load(model_vae_save_path)
        #     model_actor_para = torch.load(model_actor_save_path)
        #     model_critic_para = torch.load(model_critic_save_path)

        #     all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        #     train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        #     train_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        #     train_i2ppo(HIDDEN_LAYERS, model_vae_para, model_actor_para, model_critic_para, train_env, valid_env, args, add_str, log_dir_path)

        # elif args.ppo:
        #     all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        #     train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        #     train_env.set_env_info(S_INFO, 6, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        #     add_str = 'ppo'
        #     train_ppo_pure(None, HIDDEN_LAYERS, train_env, valid_env, args, add_str, log_dir_path)
        
        # # elif args.dppo:
        # #     all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        # #     train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        # #     train_env.set_env_info(S_INFO, 6, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        # #     add_str = 'dppo'
        # #     model_actor_para = torch.load('./saved_models/geser/Policy_geser_760.model')
        # #     train_dppo_pure(model_actor_para, HIDDEN_LAYERS, train_env, valid_env, args, add_str, log_dir_path)

        # elif args.geser:
        #     all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        #     train_env_1 = env_oracle.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        #     train_env_1.set_env_info(S_INFO, 6, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        #     add_str = 'geser'
        #     model_actor_para = train_il_pure(IMITATION_TRAIN_EPOCH, HIDDEN_LAYERS, train_env_1, valid_env, args, add_str, log_dir_path)

        #     # save models in the First stage
        #     model_save_dir = MODEL_DIR + '/' + add_str
        #     if not os.path.exists(model_save_dir):
        #         os.mkdir(model_save_dir)
        #     # command = 'rm ' + SUMMARY_DIR + add_str + '/*'
        #     # os.system(command)
        #     model_actor_save_path = model_save_dir + "/%s_%s_%d.model" %(str('Policy'), add_str, int(IMITATION_TRAIN_EPOCH))
        #     if os.path.exists(model_actor_save_path): os.system('rm ' + model_actor_save_path)
        #     torch.save(model_actor_para, model_actor_save_path)

        #     # model_actor_para = torch.load('./saved_models/geser/Policy_geser_760.model')

        #     train_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
        #     train_env.set_env_info(S_INFO, 6, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, REBUF_PENALTY_LOG, SMOOTH_PENALTY)
        #     train_ppo_pure(model_actor_para, HIDDEN_LAYERS, train_env, valid_env, args, add_str, log_dir_path)

if __name__ == '__main__':
    main()
