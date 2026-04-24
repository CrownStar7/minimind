import math, torch, torch.nn.functional as F
from torch import nn


'''
PretrainedConfig：所有预训练模型配置类的父类，把模型超参集中管理，别人拿到目录就能复现模型结构，而不是靠你手动传一堆参数

GenerationMixin：生成相关的接口，如generate()，generate_stream()，generate_with_past()，generate_with_past_and_stream()

MoeCausalLMOutputWithPast是 Transformers 里针对（MoE）CausalLM 的输出 dataclass：把 loss / logits / past_key_values / hidden_states 等按统一字段名包装起来。这样下游代码能用 outputs.loss、outputs.logits、outputs.past_key_values 这种稳定 API，而不是记你返回的是 tuple 第几个。
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # ...
        if labels is not None:
            # ...
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
很多训练循环/Trainer 只认 outputs.loss 来反传；你这里传了 labels 就会得到 loss 字段，不用你手动约定 “第 0 个返回值是 loss”。能自然写出
    outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
    logits = outputs.logits[:, -1, :] / temperature
    past_key_values = outputs.past_key_values if use_cache else None

PreTrainedModel：所有预训练模型的基类，把模型结构和权重管理统一起来，别人拿到目录就能加载模型，而不是靠你手动写一堆加载逻辑。它把“配置—模型”绑定成 HF 的标准形态；否则很多 HF 工具链（加载、保存、pipeline、trainer）就要你自己写大量 glue code。
'''
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    '''
    transformers 的 AutoConfig 会通过 model_type 自动匹配对应的配置类，
    是生态兼容的核心标识（需与模型注册时的名称一致）
    '''
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=4, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 隐藏层维度 更大的维度能捕获更复杂的语义信息，但计算成本更高。小模型通常用 512-768，大模型用 2048-8192
        self.hidden_size = hidden_size
        #Transformer 层的堆叠数量 更多层能学习更深层的抽象特征，但训练更慢。GPT-2 small 用 12 层，GPT-3 用 96 层
        self.num_hidden_layers = num_hidden_layers
        # ：use_moe=True 时，每层会有多个专家网络，动态选择激活。在不显著增加推理成本的情况下扩大模型容量
        # 路由就是线性层  self.router = nn.Linear(hidden_size, num_experts)
        # 四个专家 Token 1: 专家得分 [0.1, 0.6, 0.2, 0.1] → 选择专家 1(0.6) 和专家 2(0.2) → 归一化 [0.75, 0.25]
        # 最终输出 = 0.75 * Expert1(token1) + 0.25 * Expert2(token1)
        self.use_moe = use_moe
        # Dropout 概率（防止过拟合）：模型中随机丢弃部分神经元的比例，0.0 表示不使用 Dropout。
        self.dropout = kwargs.get("dropout", 0.0)
        # 词表大小：模型支持的唯一 token 数量（包括字符、子词等），需与词表文件（如 vocab.json）的大小一致。
        self.vocab_size = kwargs.get("vocab_size", 6400)
        # 序列开始和结束的特殊 token ID,标记文本边界，帮助模型理解序列结构
        # 具体例子：编码句子 "你好世界"
        # Tokenizer 输出: [1, 345, 678, 901, 234, 2]
        #                [BOS, 你, 好, 世, 界, EOS]
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # 是否使用 Flash Attention 优化算法,大幅降低显存占用（O(N) vs O(N²)），加速训练和推理
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        # num_key_value_heads=4，配合 num_attention_heads=8 实现 2:1 分组,减少50% KV cache 显存占用，加速推理。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # 每个注意力头的向量维度 head_dim = 512 / 8 = 64 通常 hidden_size / num_attention_heads，保证总维度一致
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # FFN 中使用的非线性激活函数 SiLU 在 Transformer 中表现优于 ReLU，Llama 系列都用 SiLU
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # FFN 中间层的维度大小  通常是 hidden_size 的 4 倍左右，这里用 π倍并对齐到 64 以优化硬件计算
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # 模型支持的最大序列长度 max_position_embeddings=32768 表示最多处理 32K tokens  限制上下文窗口大小，影响显存占用
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # 旋转位置编码的基础频率参数 更大的 theta 能更好地外推到更长序列，Llama 2 用 1000，长上下文模型用更大值
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # 是否在推理时使用 RoPE 缩放技术 允许模型处理超过训练长度的序列（如训练 2K，推理 32K）
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        """
        RoPE 缩放配置,
            type: "yarn" - 使用 YaRN 算法
            factor: 16 - 扩展 16 倍长度
            original_max_position_embeddings: 2048 - 原始训练长度
        """
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        # MoE 专用参数
        # MoE 层中的专家网络数量
        self.num_experts = kwargs.get("num_experts", 4)
        # 每个 token 激活几个专家 ：num_experts_per_tok=1 表示 top-1 路由
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        # MoE 中间层维度  每个专家的 FFN 中间层大小 可以独立调整专家网络大小
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # 归一化 top-k 概率 确保专家权重和为 1，稳定训练
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        # 路由辅助损失系数 鼓励专家负载均衡，防止某些专家被过度使用
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    '''
    稳定训练 - 通过归一化激活值，防止梯度爆炸或消失
    加速收敛 - 使模型训练更快更稳定
    提高性能 - 在 Transformer 架构中表现优异

    相比传统 LayerNorm 的优势：
        更简单 - 不需要计算均值，只计算均方根
        更快 - 计算量更少（省略了减均值的步骤）
        效果相当 - 在大多数情况下性能与 LayerNorm 相当或更好
    '''
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

'''
为什么需要位置编码
# 自注意力计算（简化版）
Q, K, V = x @ W_q, x @ W_k, x @ W_v
attention = softmax(Q @ K.T) @ V

# 问题：如果打乱输入顺序
x_shuffled = x[[2, 0, 1]]  # 重新排列
# 注意力矩阵会跟着变化，但每个 token 看到的"集合"是一样的
# 模型无法区分 "I love you" 和 "you love I"

# 加入位置信息
x_with_pos = x + positional_encoding(positions)

# 现在每个 token 携带了"我在第几个位置"的信息
# "I"(pos=0) 和 "I"(pos=2) 会有不同的表示

！！！！也就是说，通过不断反向传播，在训练一个具有位置信息的embedding之间的关系，得到一个最优解的多维函数，人类的思维，语言逻辑形成了规律。

'''
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


'''
这是 **RoPE (Rotary Position Embedding) 的应用函数**，将预计算的旋转编码应用到 Query 和 Key 上。

## 核心操作

```python
def rotate_half(x):
    # 将向量分成两半并交换位置，同时第一半取负
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
```

**例子**：
```python
x = [1, 2, 3, 4, 5, 6]
# 分成两半：[1,2,3] 和 [4,5,6]
rotate_half(x) = [-4, -5, -6, 1, 2, 3]
```
## 旋转公式
```python
q_rotated = q * cos + rotate_half(q) * sin
```
这等价于 2D 旋转矩阵：
```
[cos θ  -sin θ] [q1]   [q1*cos - q2*sin]
[sin θ   cos θ] [q2] = [q1*sin + q2*cos]
```
简单说：**通过旋转操作把位置信息"编码"进 Q 和 K，让注意力计算自动感知相对距离**。
'''
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

'''
虽计算量一样，但瓶颈不在计算速度，而在从显存中读取KV缓存速度，以往8个，现在两个，通过扩展，减少75%数据量
'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    用途：将少量 KV heads 复制以匹配更多 Q heads
    例如：8 个 Q heads，2 个 KV heads → 每个 KV 复制 4 次
    节省内存和计算，性能损失很小
    '''
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            # # 把以前缓存的 past_key_value[1] (旧的V) 和 刚算出来的 xv (新的V) 拼起来
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 为了让不同长度的句子对齐，我们会补齐一些 [PAD] 字符。这些字符不携带语义，计算注意力时应当忽略它们。
            # 如果 attention_mask 的值为 0（代表是 padding），则加一个极大的负数（$-10^9$），彻底屏蔽该位置。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # self.act_fn(self.gate_proj(x)) 门控信号mask
        # self.act_fn(self.gate_proj(x)) * self.up_proj(x)：这是关键。激活后的“门控路”与“上升路”相乘，实现特征的选择和增强。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    '''
    不再让一个巨大的 FFN处理所有任务，而是准备一堆小 FFN（专家），每次只选最懂行的几个来干活。
    如果你选了 2 个专家，他们的原始得分可能是 0.4 和 0.2。归一化后变成 0.67 和 0.33，确保输出的幅值不会因为专家数量的变化而失控。
    
    假设我们有 4个专家，设置 Top-K = 2：

    - Token A 的路由分数：专家1(0.6), 专家2(0.3), 专家3(0.1), 专家4(0.0)
        - _Top-2 选中：专家1 和 专家2_    
    - Token B 的路由分数：专家2(0.5), 专家3(0.4), 专家1(0.1), 专家4(0.0)
        - _Top-2 选中：专家2 和 专家3_
            
    执行过程：
    - 专家 1：发现 Token A 选了它。计算 `Expert1(Token A) * 0.6`，填入 `y[A]`。
    - 专家 2：发现 Token A 和 Token B 都选了它。
        - 计算 `Expert2(Token A) * 0.3` 累加到 `y[A]`。
        - 计算 `Expert2(Token B) * 0.5` 填入 `y[B]`。
    - 专家 3：发现 Token B 选了它。计算 `Expert3(Token B) * 0.4` 累加到 `y[B]`。
    - 专家 4：发现没人选它。它在 `if mask.any()` 那里就被跳过了，不参与计算。
    '''
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # freqs_cos, freqs_sin ：[32768, 96],已经生成位置表，这行代码是在切片取值
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        # 单次调用，生成一个词，在外层的model.generate()会重复调用，不停生成token
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 训练时输出维度：[Batch_Size, Seq_Length, Vocab_Size]
        # 推理时维度： 为了省资源，可能只取最后一个 Token 的 Logit，形状会变成 [Batch, 1, Vocab]
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            '''
            x 的形状变化：
            [Batch, Seq-1, Vocab]  -->  [(Batch * (Seq-1)), Vocab]
            y 的形状变化：
            [Batch, Seq-1]         -->  [(Batch * (Seq-1))]

            位置 0: 看到 [我]           → 预测 "爱"  ✓ 有目标
            位置 1: 看到 [我, 爱]       → 预测 "你"  ✓ 有目标  
            位置 2: 看到 [我, 爱, 你]   → 预测 "。"  ✓ 有目标
            位置 3: 看到 [我, 爱, 你, 。] → 预测 ???  ✗ 没有下一个词了！

            logits[..., :-1, :]  # 去掉位置 3，保留 [0,1,2]
            labels[..., 1:]      # 去掉位置 0，保留 [1,2,3]
            '''
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 在预测任务中，用第t个字预测第t+1个字。所以代码将预测值（logits）去掉最后一位，将标签（labels）去掉第一位，让它们一一对应。
            # 一次计算t-1次，然后求平均loss
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 如果想针对一个问题生成 3 个不同的答案，这里会把输入的句子复制 3 份，方便并行计算。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        # 建立一个“记事本”，记录哪些句子已经生成完了（遇到了结束符）。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 获取缓存的长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            '''
            关键切片：input_ids[:, past_len:]
            这是实现“只算最后一个字”的核心：
            第一次运行（Prefill 阶段）：
            此时 past_key_values 是 None，所以 past_len = 0。
            input_ids[:, 0:] 代表把输入的整句话都喂给模型。
            模型计算整句话的特征，并将生成的 KV 缓存存入 past_key_values。
            
            后续运行（Decoding 阶段）：
            此时 past_key_values 已经有值了，假设里面存了 10 个词的状态，past_len = 10。
            input_ids[:, 10:]：注意！如果当前 input_ids 总长是 11，那么这个切片只取索引为 10 的那一个词。
            结果：你只传了一个词进入 forward 函数。
            '''
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 取出模型对下一个词的预测概率，并除以 Temperature（温度）
            # 温度越高，分数越平均，模型越“乱说话”；温度越低，高分越集中，模型越保守
            logits = outputs.logits[:, -1, :] / temperature
            # 如果某个词之前出现过，就降低它的分数，让模型倾向于说新词。
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            # 选出得分最高的前 K 个词，把剩下的词概率全抹成负无穷。只在“尖子生”里选，防止胡言乱语
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # 按照概率从高到低加和，直到和超过 $P$（如 0.85）。只保留这前 $N$ 个词。这比 Top-K 更灵活
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # do_sample: 如果为真，从概率分布里随机抽一个（有惊喜）
            # argmax: 否则，永远选概率最大的那一个（很死板）
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 如果某句话已经结束了，那它后面就只能填 eos_token_id
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 把刚算出来的词拼到原句后面
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # 把当前算好的中间状态存起来，下一轮循环直接用
            past_key_values = outputs.past_key_values if use_cache else None
            # 每产生一个词，就立刻传给显示器打印，不用等整句写完
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                # 如果所有句子都遇到了结束符，提前下班，节省时间
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids