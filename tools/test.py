# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner._flexible_runner import FlexibleRunner

from mmrotate.utils import register_all_modules
import copy
from typing import Dict, Union
from mmengine.config import Config, ConfigDict

ConfigType = Union[Dict, Config, ConfigDict]

class EVRunner(FlexibleRunner):
    
    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'FlexibleRunner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        strategy = cfg.pop('strategy')
        runner = cls(
            model=cfg['model'],
            work_dir=cfg.get('work_dir', 'work_dirs'),
            experiment_name=cfg.get('experiment_name'),
            train_dataloader=cfg.get('train_dataloader'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            train_cfg=cfg.get('train_cfg'),
            val_dataloader=cfg.get('val_dataloader'),
            val_evaluator=cfg.get('val_evaluator'),
            val_cfg=cfg.get('val_cfg'),
            test_dataloader=cfg.get('test_dataloader'),
            test_evaluator=cfg.get('test_evaluator'),
            test_cfg=cfg.get('test_cfg'),
            strategy=strategy,
            auto_scale_lr=cfg.get('auto_scale_lr'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            cfg=cfg,
        )
        return runner
    
    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore
        dispatch_kwargs = dict(
            init_weights_for_test_or_val=self.cfg.get(
                'init_weights_for_test_or_val', True))
        self.strategy.prepare(
            self.model,
            optim_wrapper=self.optim_wrapper, 
            dispatch_kwargs=dispatch_kwargs
            )
        self.model = self.strategy.model

        self.load_or_resume()

        self.call_hook('before_run')
        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')

        return metrics
    


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--task', type=int, default=2, help='Task1 (Object Detection) or Task2 (Object Attribute Caption)')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    # register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    # cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        # define the task type
        if args.task == 1:
            cfg.test_evaluator = cfg.test_evaluator_task1
        elif args.task == 2:
            cfg.test_evaluator = cfg.test_evaluator_task2
            # add caption to test_cfg
            if 'rcnn' in cfg.model.test_cfg:
                cfg.model.test_cfg.rcnn.caption = True
            else:
                cfg.model.test_cfg.caption = True
        # update outfile_prefix
        cfg.test_evaluator.outfile_prefix = osp.join(cfg.work_dir, cfg.test_evaluator.outfile_prefix.split("/")[-1])
        

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    runner = EVRunner.from_cfg(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
