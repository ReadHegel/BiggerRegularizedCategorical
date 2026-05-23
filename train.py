import os

os.environ['MUJOCO_GL'] = 'osmesa' #'egl'

from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list
from jaxrl.agent.networks import NETWORKS

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('start_training', 5000,'Number of training steps to start training.')
flags.DEFINE_string('env_names', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform evaluations with temperature=0.')
flags.DEFINE_boolean('render', True, 'Whether to log the rendering to wandb.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
# flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.') moved to BRO_CRITIC_CONFIG
flags.DEFINE_string('arch', 'bro', 'Architecture to use.')
flags.DEFINE_integer('log_interval', 1000, 'Print progress every N env steps (use 1 for smoke).')


def _log(msg: str) -> None:
    print(msg, flush=True)

def main(_):
    from dotenv import load_dotenv
    load_dotenv()

    if FLAGS.arch not in NETWORKS:
        raise ValueError(f"Unsupported architecture: {FLAGS.arch!r} should be one of {NETWORKS}")

    _log(
        f"[train] start arch={FLAGS.arch} env={FLAGS.env_names} seed={FLAGS.seed} "
        f"max_steps={FLAGS.max_steps} start_training={FLAGS.start_training} "
        f"log_interval={FLAGS.log_interval}"
    )

    if FLAGS.log_to_wandb:
        import wandb
        wandb.init(
            config=FLAGS,
            entity=os.getenv('WANDB_ENTITY'),
            project=os.getenv('WANDB_PROJECT'),
            group=f'{FLAGS.env_names}',
            name=f'{FLAGS.seed}'
        )
        
    env_names = get_environment_list(FLAGS.env_names)
    _log(f"[train] creating env tasks={env_names}")
    env = ParallelEnv(env_names, seed=FLAGS.seed)
    if FLAGS.offline_evaluation:
        eval_env = ParallelEnv(env_names, seed=FLAGS.seed+42)
    else:
        eval_env = None
        
    eval_interval = FLAGS.eval_interval if FLAGS.offline_evaluation else 5000
        
    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['use_l2_weight_norm'] = FLAGS.arch == 'simbaV2'
    # kwargs['width_critic'] = FLAGS.width_critic
    
    num_tasks = len(env.envs)

    _log(f"[train] initializing BRC num_tasks={num_tasks} use_l2={kwargs['use_l2_weight_norm']}")
    agent = BRC(
        FLAGS.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        arch=FLAGS.arch,
        **kwargs,
    )
    _log("[train] BRC ready, starting env loop")

    batch_size = 1024 if agent.multitask else 256

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)   
    
    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)
        
    statistics_recorder = EpisodeRecorder(num_tasks)
    
    observations = env.reset()

    for i in range(1, FLAGS.max_steps + 1):
        if i == 1:
            _log("[train] step 1 (random actions until start_training)")
        if i == FLAGS.start_training:
            _log(f"[train] step {i}: training updates begin")
        elif i > FLAGS.start_training and i % FLAGS.log_interval == 0:
            _log(f"[train] step {i}/{FLAGS.max_steps}")
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0) # dlaczego tu jest temperature 1.0? TODO
        next_observations, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample(batch_size, FLAGS.updates_per_step)
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            _ = agent.update(batches, FLAGS.updates_per_step, i)
            if i % eval_interval == 0 and i >= FLAGS.start_training:  
                _log(f"[train] step {i}: eval/log")
                info_dict = statistics_recorder.log(FLAGS, agent, replay_buffer, reward_normalizer, i, eval_env, render=FLAGS.render)

    _log(f"[train] finished {FLAGS.max_steps} steps")

if __name__ == '__main__':
    app.run(main)
