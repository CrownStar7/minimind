# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

'''
PretrainedConfig 是所有预训练模型配置类的父类，提供了统一的接口（如保存 / 加载配置、验证参数合法性等），
自定义模型时需继承此类以兼容 transformers 生态（如 AutoConfig 自动加载）
'''
from transformers import PretrainedConfig

'''
1,在当前已有的所有词（语义），以及这些词的先后顺序（相对位置）共同作用下，下一个词最可能是什么
2,模型判断 “下一个词是什么” 时，必须依赖「词的顺序」—— 而 “相对位置关系” 就是模型识别 “顺序” 的唯一依据（通过位置编码实现）
'''

class MiniMindConfig(PretrainedConfig):
    '''
    transformers 的 AutoConfig 会通过 model_type 自动匹配对应的配置类，
    是生态兼容的核心标识（需与模型注册时的名称一致）
    '''
    model_type = "minimind"

    def __init__(
            self,
            # Dropout 概率（防止过拟合）：模型中随机丢弃部分神经元的比例，0.0 表示不使用 Dropout。
            dropout: float = 0.0,
            # 句首标记（Begin Of Sequence）的 token ID：文本编码时用于标识句子开头，需与词表（vocab）中的 ID 一致。
            bos_token_id: int = 1,
            # 句尾标记（End Of Sequence）的 token ID：文本编码时用于标识句子结束，需与词表一致。
            eos_token_id: int = 2,
            # 隐藏层激活函数：silu 即 Sigmoid Linear Unit（σ(x)・x），是大模型常用激活函数（比 ReLU 更平滑），支持 relu/gelu 等其他选项。
            hidden_act: str = 'silu',
            # 隐藏层维度：Transformer 编码器 / 解码器中每个 token 的向量维度（核心超参数），决定模型容量，通常为 2 的幂（如 512、1024）。
            hidden_size: int = 512,
            # Feed-Forward 网络中间层维度：Transformer 中「自注意力层后」的全连接层维度，默认 None 时通常按 hidden_size * 4 计算（大模型常见设计）。
            intermediate_size: int = None,
            # 最大序列长度：模型支持的最长输入文本长度（token 数），32768 表示支持 32k 长文本，需与位置编码（Positional Embedding）的维度匹配。
            max_position_embeddings: int = 32768,
            # 自注意力头数：将 hidden_size 拆分为多个头并行计算注意力，提升模型对不同特征的捕捉能力（需满足 hidden_size % num_attention_heads == 0，否则无法均分维度）。
            num_attention_heads: int = 8,
            # 	Transformer 隐藏层数量（即编码器 / 解码器的层数），层数越多模型拟合能力越强，但训练成本越高。
            num_hidden_layers: int = 8,
            # KV 头数（用于分组注意力 / FlashAttention）：在高效注意力机制中，将 Key/Value 投影到 fewer 个头上（如 2 个），减少计算量（需满足 num_attention_heads % num_key_value_heads == 0）。
            num_key_value_heads: int = 2,
            # 词表大小：模型支持的唯一 token 数量（包括字符、子词等），需与词表文件（如 vocab.json）的大小一致。
            vocab_size: int = 6400,
            # RMSNorm 归一化的极小值：用于避免分母为 0，RMSNorm 是大模型常用的归一化方式（比 LayerNorm 计算更高效）。
            rms_norm_eps: float = 1e-05,
            # RoPE 位置编码的 theta 参数：RoPE（Rotary Position Embedding）通过旋转矩阵注入位置信息，theta 决定位置编码的周期（值越大，周期越长，适合长文本）。
            rope_theta: int = 1000000.0,
            # 是否启用 RoPE 长度外推：推理时若输入文本长度超过 max_position_embeddings，通过缩放 RoPE 参数避免位置编码失效（如 YARN 方法）。
            inference_rope_scaling: bool = False,
            # 是否启用 FlashAttention：Facebook 提出的高效注意力实现，大幅降低显存占用和计算时间，是大模型训练 / 推理的常用优化。
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            # 是否启用 MOE 结构：True 表示模型使用混合专家架构，False 则为普通 Transformer。
            
            # 不就是多个全连接层（不同全连接层代表放大不同方面的特征），
            # 用一个网络（路由器）根据sigmiod，决定哪几个全连接层接收输入，并和通用的全连接层加权求和
            use_moe: bool = False,
            # 每个 token 选择的专家数：MOE 中每个 token 仅由 top-k 个专家处理（如 2 个），平衡性能和计算量。
            num_experts_per_tok: int = 2,
            # 可路由专家总数：模型中独立的专家网络数量（如 4 个），每个专家负责处理特定类型的 token。
            n_routed_experts: int = 4,
            # 共享专家数量：所有 token 都会经过的「共享专家」（区别于「可路由专家」），提升模型泛化能力（避免部分专家被闲置）。
            n_shared_experts: int = 1,
            # 专家选择的评分函数：计算每个 token 与专家的匹配度，softmax 会将评分归一化为概率，其他可选如 sigmoid。
            scoring_func: str = 'softmax',
            # 辅助损失的权重：MOE 中为避免「专家闲置」（部分专家几乎不被选择），添加辅助损失（如专家均衡损失），alpha 控制辅助损失在总损失中的占比。
            aux_loss_alpha: float = 0.1,
            # 是否在序列级别计算辅助损失：True 表示基于整个序列的专家选择情况计算均衡损失，False 则按单个 token 计算。
            seq_aux: bool = True,
            # 是否标准化 top-k 专家的概率：True 会将选中的 k 个专家的概率重新归一化，确保概率和为 1，提升稳定性。
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        # 修改模型对 “位置信息” 的处理方式，让模型误以为 “超长文本的位置” 仍在自己熟悉的 “训练窗口” 内，从而正常理解内容

        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# Python 内置的数学工具库，提供基础数学运算支持
import math
# PyTorch 深度学习框架的核心库，是所有模型训练 / 推理的基础,提供多种参数初始化方法，避免模型训练时因参数初始值不当导致的梯度消失 / 爆炸
# 提供张量（Tensor）操作、自动求导（Autograd）、GPU 加速、神经网络模块（nn）等核心能力，后续所有模型的参数（权重）、输入数据都以 PyTorch 张量形式存储和处理。
import torch
# PyTorch 中神经网络参数的初始化工具模块，缩写为 init（约定俗成的简写，方便调用）
import torch.nn.init as init
# 提供无状态的神经网络操作（即操作本身不存储参数，仅接收输入和参数计算输出）
# 调用激活函数（F.silu、F.gelu）、损失函数（F.cross_entropy、F.mse_loss）
import torch.nn.functional as F
# 提供封装好的可训练模块（类），这些模块会自动管理内部参数（权重、偏置），支持自动求导和参数优化。
# 定义模型层结构，例如全连接层（nn.Linear）、归一化层（nn.RMSNorm）、Dropout 层（nn.Dropout）、Embedding 层（nn.Embedding）等，后续构建 Transformer/MOE 模型的层都会依赖 nn 模块。
from torch import nn
# 当配置中指定 hidden_act='silu' 时，可通过 ACT2FN[config.hidden_act] 直接获取对应的激活函数（无需手动判断字符串对应的函数）
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
# 从 transformers 库导入构建自定义模型的核心基类，是兼容 transformers 生态的关键。
# PreTrainedModel: 所有预训练模型的父类，提供模型加载 / 保存（from_pretrained/save_pretrained）、设备迁移（to(device)）、参数冻结等通用功能，自定义模型（如 MiniMindModel）需继承此类。
# GenerationMixin： 生成式模型的混入类（Mixin），提供文本生成的核心方法（如 generate()），包含贪心搜索、束搜索（Beam Search）等生成策略，让自定义模型无需手动实现生成逻辑。
# 之前讲解过的配置类父类，此处导入是为了在模型类中接收配置实例（如 __init__(self, config: PretrainedConfig)），确保模型与配置的联动。
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
'''
统一生成式模型的输出格式，将模型的核心输出（预测 logits、隐藏状态、注意力权重等）封装成一个具名元组（Named Tuple），方便后续调用（如计算损失、生成文本时获取中间结果）。
输出类包含的关键属性（按需使用）：
logits: 模型最终预测的 token 概率分布（shape: [batch_size, seq_len, vocab_size]），用于计算损失或采样下一个 token；
past_key_values: 缓存的注意力层 Key/Value 张量，用于增量生成（避免重复计算已生成 token 的注意力，提升生成速度）；
hidden_states: 模型各层的隐藏状态（可选），用于特征提取或调试；
attentions: 各注意力层的注意力权重（可选），用于可视化或分析模型注意力分布。
'''
from transformers.modeling_outputs import CausalLMOutputWithPast

'''
为什么继承 nn.Module：
获得 PyTorch 内置的参数管理（如 nn.Parameter 自动注册为可训练参数）、设备迁移（to(device)）、前向传播接口（forward 方法）等核心功能；
确保该层能像 nn.Linear、nn.Dropout 一样，嵌入到完整的神经网络中使用。

通过 “标准化 + 自适应缩放”，让模型中每个 token 的隐藏向量幅度保持一致，避免因数值过大 / 过小导致的训练不稳定（如梯度消失 / 爆炸）。
相比传统 LayerNorm，RMSNorm 少了「减均值」的步骤，计算量更小、显存占用更低，是大模型（如 LLaMA、GPT-4）的主流选择。
在 Transformer 层（或 MOE 专家层）的输入 / 输出处插入该层，传入 eps=config.rms_norm_eps 即可
'''
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    # 定义核心的归一化计算逻辑，用下划线 _ 开头表示「内部辅助函数」（不建议外部直接调用，仅在 forward 中使用）。
    '''
    torch.rsqrt: 计算平方根的倒数
    x.pow(2): 计算输入张量每个位置的平方（如 x=3 变成 9，x=-2 变成 4），目的是消除正负号影响，聚焦数值大小
    x.pow(2).mean(-1, keepdim=True) :-1 表示「最后一个维度」（即 dim 维度），keepdim=True 表示保持维度不变（输入 [32,128,512] → 输出 [32,128,1]），避免广播计算出错；结果是每个 token 512 维向量的「平方均值」（衡量该 token 向量的整体幅度）
    x * ... 用输入张量 x 乘以归一化系数，最终得到「均值为 0、方差近似为 1」的标准化向量（消除不同 token 向量幅度差异的影响）
    '''
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    # 定义数据的前向流动逻辑
    # .type_as(x) 避免类型不匹配（如输入是 float16，标准化后仍转回 float16，保证后续计算兼容）
    # self.weight * ... 用可训练的 weight（shape [dim]）对标准化向量的每个维度逐元素相乘（广播机制），实现 “自适应缩放”
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

'''
Yarn 长度外推

第一步：周期映射公式
p' = p mod L + s * (L/K)   s=floor(p/L)

p mod L :保证始终在一个周期内
s * (L/K): 区分不同周期中相同位置
----------------------
拆解 1：p mod L（取余运算）→ 对应 “周期内的基础位置”。比如 L=2000，p=10000 时，10000 mod 2000 = 0，即第 10000 个 token 对应 “第 5 个周期的第 0 个位置”（类似 “第 5 轮的第 1 本书”）；
✅ 生活对应：管理员不用记 “第 10000 本”，只记 “这是当前周期的第 0 本”，和第 1 个周期的 “第 0 本”（即第 1 本书）编码逻辑一致，不会懵。
拆解 2：s = floor(p / L)（向下取整）→ 对应 “周期序号”。p=10000 时，s=10000/2000=5，即第 5 个周期；
拆解 3：K 是 “周期分组数”（YARN 预设的超参数，比如 4）→ 给不同周期加 “轻微区分”，避免模型混淆 “第 1 周期的第 0 本” 和 “第 5 周期的第 0 本”，但区分度很小，不影响模型对 “周期内顺序” 的识别。
（2）相对位置补充（公式隐含逻辑）
传统编码只看 p（绝对位置），YARN 通过周期映射后，两个 token 的「相对位置」可以通过 |p1' - p2'| 计算（比如第 10000 个 token 的 p1'=0，第 10001 个 token 的 p2'=1，相对位置是 1，即 “相邻”）。✅ 生活对应：管理员通过 p' 的差值，直接知道 “两本书的前后关系”，不用管它们在哪个周期。

第二步：动态上下文压缩
（1）压缩公式
把L个token，分成M组，每组得到每个token的重要性，加权求和，得到每个代表性的代表性（共M个）
L - M = N， 会在增加N个
--------------------
YARN 会通过「注意力池化（Attention Pooling）」对早期上下文进行压缩，得到 M 个核心语义向量 H_compressed：\(H_{compressed} = \text{AttentionPool}(H, W) = \sum_{i=1}^L \alpha_i \cdot h_i\)拆解 1：α_i = softmax(W · h_i)（注意力权重）→ α_i 是第 i 个 token 的 “重要性得分”，总和为 1（比如重要的 token 得分 α_i=0.01，冗余的 token 得分 α_i=0.0001）；
✅ 生活对应：管理员判断 “第 3 本书讲核心原理（α_i 高），第 5 本书是重复举例（α_i 低）”，重点记前者。

拆解 2：sum(α_i · h_i) → 加权求和，把 L 个语义向量 “浓缩” 成 M 个（YARN 会分 M 组计算，每组对应一个核心向量），相当于 “把 1000 本书的重点，提炼成 1 条笔记”；
✅ 生活对应：管理员不记每本书的逐字内容，只记 “这 1000 本书的核心是 XX”，笔记体积小（占空间少）但信息密度高。

（2）窗口更新逻辑（公式隐含流程）当新的 N 个 token 进来（比如新的 1000 本书），窗口更新公式为：\(H_{new} = [H_{compressed}, h_{L+1}, h_{L+2}, ..., h_{L+N}]\)拆解：把压缩后的 M 个核心向量（笔记本笔记），和新的 N 个 token 语义向量（新摆上桌面的书）拼接，总长度仍为 L（M + N = L），既没超出窗口，又保留了前文重点；
✅ 生活对应：桌面始终保持 2000 个 “信息单元”（1000 条笔记 + 1000 本新书），不会溢出，且笔记能替代原书的核心信息。


Yarn-->RoPE->embedding-->word2verc
为了让模型处理文本，需要将文本数值化，除了one-hot这种方式外，但这会有唯独灾难（有大量的无效的0，且不能表示语义），但如果用一种稠密向量表示单个词语(token)，让相似语义的词语(token)向量夹角较小，相同语义（语义较强的，模长的大，语义小的，模长小），
但怎么做到呢？方法就是使用对比损失函数，训练模型，强迫模型将语义相似的，向量夹角相近。具体过程如下：
我们用 SimCSE 训练句向量 的场景做实际训练例子 —— 这是对比学习（InfoNCE Loss）最经典的应用，全程还原 “数据准备→模型计算→损失优化→向量收敛” 的完整过程，每个步骤都对应公式，直观看到损失函数如何 “逼着” 向量满足语义约束。
训练完毕后，相当于相同语义的，向量在同一附近，实际跟人的思想有点像，谈一个话题，相同语义的多个词，相继出现的概率大。
模型不理解现实的意思，他只是明白根据前面的多个向量，向量的顺序，下一个向量应该是这个，然后输出，转换为人理解的词语，看着大模型似乎理解了，我的意思，实际上不是，只是那个向量被计算出是下一个向量的概率大。

训练过程：12



一、训练任务定义
目标：训练模型让「同义句向量夹角接近 0°，非同义句向量夹角接近 180°」。选用 3 个句子作为训练样本（模拟海量训练数据中的一个批次）：
锚点句（x）：“我爱吃苹果”（核心语义：苹果相关）
正样本句（x⁺）：“我喜欢吃苹果”（和锚点同义，语义相似）
负样本句（x⁻₁, x⁻₂）：“我爱吃汽车”“石头很硬”（和锚点不同义，语义不相似）
模型：用简化版 BERT（仅保留编码器和句向量输出层），句向量维度为 2（方便计算和可视化，实际是 768 维，逻辑完全一致）。超参数：温度 τ=0.1（控制区分度，主流取值）。
二、Step1：数据准备与向量初始化
训练刚开始时，模型的句向量是 随机初始化 的（完全不符合语义），我们先记录初始向量（随机生成合理范围的数值）：
锚点向量 x = [0.2, 0.3]ᵀ（长度∥x∥=√(0.2²+0.3²)≈0.36）
正样本向量 x⁺ = [0.5, 0.1]ᵀ（长度∥x⁺∥=√(0.5²+0.1²)≈0.51）
负样本向量 x⁻₁ = [0.7, 0.8]ᵀ（长度∥x⁻₁∥=√(0.7²+0.8²)≈1.06）
负样本向量 x⁻₂ = [0.1, 0.9]ᵀ（长度∥x⁻₂∥=√(0.1²+0.9²)≈0.91）
此时向量完全混乱：比如正样本 x⁺和锚点 x 的夹角很大，负样本 x⁻₁和 x 的夹角很小 —— 损失函数会捕捉到这种 “语义不匹配”，并触发优化。
三、Step2：计算 InfoNCE Loss（核心公式应用）
根据 InfoNCE Loss 公式，分 3 步计算损失：
1. 计算所有样本对的夹角余弦值（cosθ）
关键公式：cosθₓᵧ = (x・y)/(∥x∥・∥y∥)（点积 ÷ 长度乘积）
正样本对（x, x⁺）：x・x⁺ = 0.2×0.5 + 0.3×0.1 = 0.1 + 0.03 = 0.13cosθₓₓ⁺ = 0.13/(0.36×0.51) ≈ 0.13/0.18 ≈ 0.722（夹角≈43°，太大，不符合 “同义句近”）
负样本对（x, x⁻₁）：x・x⁻₁ = 0.2×0.7 + 0.3×0.8 = 0.14 + 0.24 = 0.38cosθₓₓ⁻¹ = 0.38/(0.36×1.06) ≈ 0.38/0.38 ≈ 1.0（夹角≈0°，太小，不符合 “非同义句远”）
负样本对（x, x⁻₂）：x・x⁻₂ = 0.2×0.1 + 0.3×0.9 = 0.02 + 0.27 = 0.29cosθₓₓ⁻² = 0.29/(0.36×0.91) ≈ 0.29/0.33 ≈ 0.879（夹角≈28°，太小，不符合约束）
2. 计算分子和分母（公式核心项）
分子：exp (cosθₓₓ⁺ / τ) = exp (0.722 / 0.1) = exp (7.22) ≈ 1360分母：分子 + sum (exp (cosθₓₓ⁻ᵢ / τ)) = 1360 + exp (1.0/0.1) + exp (0.879/0.1)= 1360 + exp(10) + exp(8.79) ≈ 1360 + 22026 + 7350 ≈ 30736
3. 计算最终损失
L = -log (分子 / 分母) = -log (1360/30736) ≈ -log (0.044) ≈ 3.13（损失值很大，说明向量严重不符合语义约束）
四、Step3：反向传播优化（损失函数逼着向量调整）
模型的目标是 “最小化损失 L”，通过 反向传播 调整向量的每个分量（0.2、0.3、0.5 等数值），调整方向完全由 InfoNCE Loss 的梯度决定：
对正样本 x⁺：要让 cosθₓₓ⁺增大（接近 1）→ 调整 x⁺的分量，让它和 x 的方向更接近（比如 x⁺从 [0.5,0.1]→[0.3,0.4]，和 x=[0.2,0.3] 方向趋同）；
对负样本 x⁻₁、x⁻₂：要让 cosθₓₓ⁻ᵢ减小（接近 - 1）→ 调整 x⁻₁、x⁻₂的分量，让它们和 x 的方向相反（比如 x⁻₁从 [0.7,0.8]→[-0.3,-0.4]，和 x 方向相反）；
对锚点 x：微调分量，让它和 x⁺的方向更一致，同时和 x⁻₁、x⁻₂的方向更相反。
这个过程会 反复迭代（比如训练 1000 轮），每一轮都重新计算损失、调整向量，直到损失降到最低。
五、Step4：训练收敛（向量满足语义约束）
经过多轮迭代后，损失 L 从 3.13 降到 0.01（接近最小值），此时的向量完全符合语义逻辑：
锚点向量 x = [3, 4]ᵀ（长度∥x∥=5，核心语义：苹果）
正样本向量 x⁺ = [6, 8]ᵀ（长度∥x⁺∥=10，是 x 的 2 倍，方向完全相同）
负样本向量 x⁻₁ = [-3, -4]ᵀ（长度∥x⁻₁∥=5，和 x 方向完全相反）
负样本向量 x⁻₂ = [-6, -8]ᵀ（长度∥x⁻₂∥=10，和 x 方向完全相反）
验证：重新计算损失（符合约束）
计算 cosθ：
cosθₓₓ⁺ = (3×6 + 4×8)/(5×10) = (18+32)/50 = 50/50 = 1.0（夹角 0°，同义句近）
cosθₓₓ⁻¹ = (3×(-3) + 4×(-4))/(5×5) = (-9-16)/25 = -25/25 = -1.0（夹角 180°，非同义句远）
cosθₓₓ⁻² = (3×(-6) + 4×(-8))/(5×10) = (-18-32)/50 = -50/50 = -1.0（夹角 180°，符合约束）
计算损失：分子 = exp (1.0/0.1) = exp (10) ≈ 22026分母 = 22026 + exp (-1.0/0.1) + exp (-1.0/0.1) = 22026 + 2×exp (-10) ≈ 22026（exp (-10)≈4.5e-5，可忽略）L = -log (22026/22026) = -log (1) = 0（损失最小，向量完全满足语义约束）
六、训练结果的核心意义
向量关系匹配语义：
同义句（x 和 x⁺）：方向相同（夹角 0°），长度不同（x⁺更长，代表语义强度更强）；
非同义句（x 和 x⁻₁、x⁻₂）：方向相反（夹角 180°），长度不影响语义差异。
损失函数的作用：
整个过程中，InfoNCE Loss 是 “指挥棒”—— 通过 “惩罚不符合语义的向量关系”（初始损失大），逼着模型调整向量，最终让 “向量夹角” 完美匹配 “语义相似度”。
实际应用价值：
训练好后，给模型输入 “我爱吃红苹果”（新的苹果相关句子），它会输出和 x 方向接近的向量（夹角小）；输入 “电脑很好用”（无关句子），会输出和 x 方向相反的向量（夹角大）—— 这就是 “夹角代表语义相似度” 的来源。
总结（实际训练的核心逻辑）
对比学习（InfoNCE Loss）的实际训练，就是 “随机向量→计算损失（捕捉语义不匹配）→反向传播调整向量→损失最小（向量匹配语义）” 的循环。我们举的 2 维向量例子，和实际大模型 768 维向量的训练逻辑完全一致 —— 损失函数通过数学约束，把 “同义近、非同义远” 的语义逻辑，刻进了向量的几何关系里。
'''
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # # yarn长度外推
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
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

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
