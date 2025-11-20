import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            '''
            目标: 计算模型每个位置的预测误差，最终得到一个可反向传播的损失值，用于更新模型参数（比如 v_proj、q_proj 等层的权重）
            res.logits: (batch_size, seq_len, vocab_size)
            Y: (batch_size, seq_len)
            
            1.展平预测结果，得到一个二维张量，维度为 (batch_size, seq_len)-->展平后形状：(bsz×seq_len, vocab_size)（即 (8, 6400)）
            res.logits.view(-1, res.logits.size(-1)) —— 调整预测结果形状
            CrossEntropyLoss 要求输入是 2 维张量 (num_samples, num_classes)（num_samples 是样本数，num_classes 是类别数）—— 这里把每个 token 的预测都当成一个独立的 “分类样本”（预测该 token 对应的下一个 token 是 6400 类中的哪一类）。

            2.Y.view(-1) —— 调整真实标签形状
            Y.view(-1)：将张量展平为 1 维张量（(bsz×seq_len,)），比如 (2,4) → (8,)；
            CrossEntropyLoss 要求真实标签是 1 维张量 (num_samples,)，每个元素是对应样本的 “真实类别 ID”（这里就是真实 token 的 ID，范围 0~6399）
            
            展平后的预测结果 (8,6400) 中，第 i 行对应 “第 i 个 token 的预测”，真实标签 (8,) 中第 i 个元素对应 “第 i 个 token 的真实下一个 token ID”
            
            .view(Y.size()) —— 还原损失形状


            具体例子
            用具体实例强化理解
                假设：
                bsz=2（2 个样本），seq_len=4（每个样本 4 个 token），vocab_size=6400；
                原始 res.logits 形状：(2,4,6400) → 展平后：(8,6400)；
                原始 Y 形状：(2,4) → 比如 [[3,5,7,9], [2,4,6,8]] → 展平后：(8,) → [3,5,7,9,2,4,6,8]；
                loss_fct 计算后（reduction="none"）：得到 8 个损失值 → [0.1, 0.3, 0.2, 0.4, 0.15, 0.25, 0.35, 0.45]；
                还原形状 .view(Y.size()) → (2,4)
            '''
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            '''
            加权平均损失（屏蔽无效位置）
            loss_mask：损失掩码，形状和 loss 一致（(bsz, seq_len)），值为 1 表示 “有效 token”（需要计算损失），0 表示 “无效 token”（如 padding 填充、特殊 token，无需计算损失）；
            逻辑：用 loss_mask 过滤掉无效位置的损失，只计算有效位置的平均损失；
            举例：假设 loss = [[0.1, 0.3, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]]（后两个 0 是 padding 位置），loss_mask = [[1,1,0,0], [1,1,0,0]]，则：
            有效损失和：(0.1×1 + 0.3×1) + (0.2×1 + 0.4×1) = 1.0；
            有效 token 数：1+1+1+1=4；
            最终损失：1.0 / 4 = 0.25（避免 padding 位置的 0 污染损失）。
            '''
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 主损失（预测 token 的损失）+ 辅助损失，让模型在学习主任务的同时，兼顾其他优化目标（比如 MOE 模型的 “负载均衡”）。
            # res.aux_loss：辅助损失（比如 MOE 模型的门控损失、对比学习损失等）
            loss += res.aux_loss

            # 一个 batch 放 128 条文本就会占满显存。但 “大 batch 训练” 能让模型收敛更稳定（梯度更平滑）
            # 把 N 个小 batch 的梯度 “攒起来”，当成 1 个大 batch 的梯度去更新参数 —— 既用了大 batch 的效果，又不用一次性存 N 个 batch 的显存
            # args.accumulation_steps：梯度累积步数（比如设为 4），表示 “累积 4 个 batch 的梯度后，再更新一次参数”；
            # 为什么是所有梯度的平均值,而不是之和,因为在计算loss时,是计算平均loss,在进行求梯度时,是平均梯度;
            # 如果不是,那么loss和batch大小就强相关,batch越大,梯度越大,lr 需要随 batch 调整,损失没有可比性,训练不稳定,所以深度学习统一用均值 loss.
            loss = loss / args.accumulation_steps
        
        # 损失放大→梯度放大，避免下溢
        # 自动求导！梯度存到 param.grad 里,loss扩大几倍,梯度就扩大几倍
        # 溢出问题: 假如此时梯度过大或者经过放大后溢出,FP16会变成inf,unscale后也是inf,那么scale的处理办法是跳过这个batch,下一次缩小放大倍数
        scaler.scale(loss).backward()


        if (step + 1) % args.accumulation_steps == 0:
            # 还原梯度（把放大的梯度缩回去） 梯度 = 放大后的梯度 / 缩放系数
            scaler.unscale_(optimizer)
            # 梯度裁剪（防止梯度爆炸）
            # 深层模型（如 Transformer）的梯度可能会 “越传越大”（梯度爆炸），比如梯度值变成 100，optimizer.step() 会让参数更新幅度过大，模型直接发散（生成乱码）
            # 训练参数（比如 q_proj.weight）本身是一个张量，它的梯度（param.grad）也是一个同形状的向量（比如 q_proj.weight 是 (512, 512)，梯度也是 (512, 512)，可看作一个长度为 512×512=262144 的向量）
            # 总范数就是把所有这些向量 “拉成一维、拼在一起”，然后计算这个 “超级长向量” 的 “长度”
            # 梯度裁剪 = 规定 “全军总兵力不能超过 10 万”—— 若超过，所有部队按同一比例缩编（比如总兵力 20 万，就缩编为原来的 1/2），既保留了各部队的兵力比例（梯度方向不变），又控制了总规模（避免梯度爆炸）
            # 总范数”（可以理解为梯度的 “总大小”）不超过 args.grad_clip,若超过，按比例缩小所有梯度。
            # 举例：梯度总范数 = 2.0，grad_clip=1.0 → 所有梯度都乘以 0.5，总范数变成 1.0，既保留了梯度的方向，又限制了幅度。
            # 避免训练震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行参数更新！根据优化器的规则（如 AdamW 的动量、权重衰减），用 param.grad 调整参数。核心公式是 param = param - lr * grad（lr 是学习率，控制调整幅度）
            # scaler.step() 会检查梯度是否溢出：若溢出，跳过本次更新（不执行 optimizer.step()），避免模型参数变成无穷大；若正常，执行更新
            # 比直接用 optimizer.step() 更安全。
            scaler.step(optimizer)
            # 动态调整缩放系数, 若出现梯度下溢，下次放大更多；若溢出，下次放大更少
            scaler.update()
            '''
            梯度会默认累积！如果不清零，下一个 batch 的 loss.backward() 会把新梯度加到旧梯度上（param.grad += 新梯度），导致梯度混乱，参数更新错误
            比如：第 1 个 batch 梯度是 0.5，第 2 个 batch 梯度是 0.3，不清零的话，优化器会用 0.8 去更新参数，完全不符合 “每个 batch 独立计算梯度” 的逻辑。
            '''
            optimizer.zero_grad(set_to_none=True)

        # 每 log_interval 步记录一次（比如 log_interval=100，就是每训练 100 个 batch 输出一次日志）
        # step == iters - 1：训练到当前 epoch 的最后一个 batch 时，强制记录一次（确保 epoch 结束有完整日志）
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 这里乘回来还原 “真实的平均损失”（比如累积 4 步，当前 loss 是 0.2，真实损失就是 0.8）
            current_loss = loss.item() * args.accumulation_steps
            # 当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            #  “已训练时间 ÷ 已训练步数 × 总步数” 算出总预计时间，再减去已训练时间，最后转成分钟
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})
        
        # step % args.save_interval == 0：每 save_interval 步保存一次
        # step == iters - 1：当前 epoch 最后一个 batch 时，强制保存一次
        # is_main_process()：分布式训练时，只让主进程保存（避免多个进程重复保存，浪费磁盘空间）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换模型到评估模式
            # 作用：暂时关闭模型的训练相关功能（如 Dropout 层、BatchNorm 层的更新），避免保存模型时，这些层的状态被干扰；
            # 后续会用 model.train() 切回训练模式，不影响后续训练。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 分布式训练兼容：如果模型用了 DistributedDataParallel（多 GPU 分布式训练），模型权重存在 model.module 里，所以要通过 model.module.state_dict() 获取（否则会保存分布式相关的冗余参数）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                # state_dict本质是一个字典：{参数名: 参数值}（比如 q_proj.weight: 张量数据、lm_head.weight: 张量数据）
                state_dict = model.state_dict()
            # v.half()：把参数从 FP32（4 字节）转成 FP16（2 字节）
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            # 保存权重文件：torch.save(state_dict, ckp)
            torch.save(state_dict, ckp)
            # 前面保存的 .pth 文件只包含模型权重，而 lm_checkpoint 会保存 完整的训练状态
            # 模型权重（state_dict）
            # 优化器状态（optimizer.state_dict()）：比如 AdamW 的动量、权重衰减参数，确保断点续训时优化器能继续按之前的状态更新
            # 混合精度训练的 scaler 状态（scaler.state_dict()）：确保续训时 scaler 的放大系数不变
            # 当前 epoch 和 step：续训时能从当前进度继续，不用从头开始
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 切回训练模式
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    '''
    根据运行设备（CPU/GPU），统一创建 “自动混合精度训练（AMP）的上下文管理器”
    下面拆解开通俗解释，先明确关键概念：
    1.上下文管理器：用 with 关键字触发，进入 with 块时执行初始化（比如启用 AMP），退出时执行清理（比如恢复默认精度），不用手动开关，很方便；
    2.自动混合精度（AMP）：PyTorch 的 torch.cuda.amp 模块，核心是 “用半精度（FP16）做计算，用全精度（FP32）存梯度”—— 既能减少 GPU 显存占用、加快训练速度，又能避免精度丢失，仅支持 NVIDIA GPU（CPU 不支持）；
    3.nullcontext：Python 内置的 “空上下文管理器”，进入 / 退出时啥也不做，纯粹是为了让 CPU 模式下的代码结构和 GPU 模式保持一致，不用写 if-else 分支。
    
    最终效果：
    无论用 CPU 还是 GPU，后续都能统一用下面的代码结构：
        with autocast_ctx:
        # 训练步骤：前向传播、计算损失等
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    如果不用 nullcontext，你可能需要写两次重复代码：
    # 不优雅的写法：用 if-else 拆分代码
    if device_type == "cpu":
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    else:
        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    用了 autocast_ctx 后，直接用一个 with 块就能覆盖两种场景，代码更简洁、易维护。
    '''

    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # PyTorch 分布式训练中的采样器选择逻辑，核心作用是：根据是否初始化了分布式环境，决定是否使用「分布式专用采样器（DistributedSampler）」，确保多 GPU 训练时数据不重复、不遗漏，且负载均衡。
    # 判断当前是否处于「分布式训练环境」（返回 True/False）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    '''
    混合精度训练的逻辑是：用 FP16 水桶做计算（省空间、速度快），用 FP32 水桶存梯度（安全、不溢出）—— 但两个水桶之间倒数据时，会出现 “溢出风险”。
    设计初衷：解决 “显存不够 + 训练太慢” 的痛点
        - 全精度（FP32）：每个参数占 4 字节，显存占用高，计算速度慢；
        - 半精度（FP16）：每个参数占 2 字节，显存占用减半，计算速度翻倍（GPU 对 FP16 优化更好）。
    半精度有个致命问题：数值范围太小（-65504 ~ 65504），梯度值很容易变成 0（梯度下溢） —— 比如模型训练后期，梯度值是 1e-8，FP16 无法表示，直接变成 0，模型无法更新（相当于 “学不动了”）
    scaler 的核心思路：用一个 “缩放器” 把损失放大 N 倍，让梯度也跟着放大 N 倍，避免下溢；更新参数前再把梯度缩回去，不影响最终结果。
    '''
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
