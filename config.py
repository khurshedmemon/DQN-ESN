class AgentConfig(object):
    is_train = True
    scale = 50000
    step = 3
    num_hidden_layer = 3    
    #max_step = 120 * scale
    terminal_round = 7
    max_step = terminal_round * scale
    testing_episode = 1000

    #memory_size = 10 * scale
    memory_size = max_step * 0.5

    batch_size = 32
    discount = 0.99
    #target_q_update_step = 1 * scale
    #added by Ali
    target_q_update_step = terminal_round * 100
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1.
    ep_decay = 0.995
    ep_end_t = memory_size
    '''
    train_frequency = 4
    learn_start = 1 * scale
    '''
    #added by Ali
    train_frequency = 1
    learn_start = 10 * terminal_round

    double_q = True
    dueling = False

    _test_step = terminal_round * 100
    _save_step = max_step
    gpu = 0


class EnvironmentConfig(object):
    env_name = 'bitcoinalpha'
    opponent = 'degree'
    mv = '0_1'  #model version
    dimensions = 16        #Number of dimensions. Default is 48
    diff_model = 'transcient'
    st_type = 'hcf'
    use_bm = False
    bm_dir = ""    

class DQNConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(FLAGS):
    config = DQNConfig
    '''
    for k, v in FLAGS.__dict__['__flags'].items():
        if hasattr(config, k):
            setattr(config, k, v)
    '''
    #added by ali
    for k in FLAGS:
        v = FLAGS[k].value
        if hasattr(config, k):
            setattr(config, k, v)

    return config
