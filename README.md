# mesh-sparse-octree

稀疏八叉树空间索引的最小复现仓库：Morton 编码、子节点索引、半开 AABB 范围查询、
兄弟叶合并、结构校验、批量构建。纯标准库（`math`）+ pytest，离线、无随机，
8 个函数全是未实现的桩，15 个测试初始全红。

## 题面

场景换成稀疏八叉树后，相邻物体的包围盒顺序乱跳、边界点还查不到：`morton_encode` 的位交错把 x、y、z 的顺序写反，x 上相邻格子的编码差成 4 或 28 而不是 1，Morton 局部性直接丢光；`morton_decode` 的位掩码少了一位，最高位被吃掉，`8**d - 1` 这种最大编码解出来的坐标只有一半，encode/decode 往返对不上；`child_index` 的中心面判据写成严格不等号（`<` / `>`）而不是 `<=` / `>=`，坐标正好压在中心面上的点被划进低半区，同一个点同时落进两个子节点的管辖范围；`insert` 的细分深度用一个全局计数器累加，先插的分支细到 max_depth、后插的分支浅一层，同一棵树的叶子深度参差不齐；`query_box` 的点判定用闭区间，压在上边界上的点被返回，边界点还会被相邻两个格子重复计入；`query_box` 只遍历叶节点、不读内部节点的 points，合并上提过的值一个都查不到；`merge_leaves` 只比较值不比较子节点数，三个同值兄弟就把父节点压成叶子，另外五个空位被一起吞掉；`validate_tree` 不检查子节点是否越出父节点边界，越界子树照样返回空错误列表；`octree_batch` 复用同一个根节点，第二批点集直接挂到第一批树上，两批数据互相污染；`insert` 对同一点重复插入不更新值，后写的值永远丢失。难点是编码与边界：Morton 位交错、子索引半边、范围查询半开区间。octree/node.py 是核心，tests/test_node.py 15个红。要修的是：上述 10 个缺陷，契约已经写在 8 个桩函数的 docstring 里，只实现函数体，不要改签名和返回形状，修完这 15 个测试应当全绿。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始失败环境的预期输出（末尾一行）：

```text
15 failed in 0.15s
```

必须是**恰好 15 个 failed**，并且没有 collection error / import error / syntax error ——
桩函数里的 `NotImplementedError` 只会在测试调用它们时抛出，不会在 conftest 或模块导入
阶段抛出。实现完 8 个函数体后，同样的命令应当变成 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_node.py` 就是验收规格：15 个测试逐条对应上面 10 个缺陷（每个缺陷 1 条）
以及 5 条组合/边界约束（编码往返、Morton 局部性、100 点暴力对照、合并前后查询等价、
批量与单独构建等价）。**测试文件不可修改**，也不要改 `conftest.py` 里把仓库根加进
`sys.path` 的那几行；要改的只有 `octree/node.py` 里 8 个函数的函数体。

## 仓库结构

```text
mesh-sparse-octree/
  README.md                 本文件
  conftest.py               把仓库根加进 sys.path（不定义 fixture）
  .gitignore                三行：__pycache__/、*.pyc、.pytest_cache/
  octree/__init__.py        导出 8 个函数
  octree/node.py            8 个桩，契约写在 docstring 里
  tests/test_node.py        15 个测试（不可修改）
```

节点用普通 dict 表示，形状固定（详见 `octree/node.py` 模块 docstring）：

```python
{
    "depth": 0,          # 根为 0，子节点 = 父节点 depth + 1
    "x": 0, "y": 0, "z": 0,   # 该深度网格下的格子坐标，0 <= x,y,z < 2**depth
    "max_depth": 6,      # 仅根节点必须携带
    "points": {},        # {point: value}，point 是三元 float 组
    "children": {},      # {child_index: node}，child_index 取 0..7
}
```

坐标定义在 `[0, 1)³` 上，区间左闭右开；深度 d 的格子边长 `2**-d`；深度上限 10。
子节点索引按位定义：`bit0 = x`、`bit1 = y`、`bit2 = z`。浮点比较容差 1e-9。

## 缺陷与测试对照

| # | 缺陷 | 测试 |
| --- | --- | --- |
| 1 | `morton_encode` 位交错把 x、y、z 顺序写反，相邻格子编码不连续 | `test_morton_encode_interleaves_x_at_low_bit` |
| 2 | `morton_decode` 位掩码少一位，最大坐标解错、往返失败 | `test_morton_decode_keeps_top_bit_for_largest_cell` |
| 3 | `child_index` 用 `<` / `>` 而不是 `<=` / `>=`，中心面上的点归错子节点 | `test_child_index_puts_center_face_point_in_high_half` |
| 4 | `insert` 细分深度用全局计数，不同分支深度不一致 | `test_insert_reaches_max_depth_on_every_branch` |
| 5 | `query_box` 用闭区间，上边界点被返回/重复计入 | `test_query_box_is_half_open_on_upper_boundary` |
| 6 | `query_box` 只查叶节点，内部节点上的值丢失 | `test_query_box_reads_values_stored_on_internal_nodes` |
| 7 | `merge_leaves` 只比较值不比较子节点数，误合并 | `test_merge_leaves_requires_eight_leaf_siblings_with_equal_values` |
| 8 | `validate_tree` 不检查子节点是否越父节点边界 | `test_validate_tree_reports_child_outside_parent_cell` |
| 9 | `octree_batch` 复用根节点，第二批点集挂到第一批树上 | `test_octree_batch_builds_independent_trees_per_batch` |
| 10 | `insert` 对同一点重复插入不更新值，后写失败 | `test_insert_duplicate_point_overwrites_value` |

组合/边界 5 条：`test_morton_encode_decode_round_trip_within_tolerance`、
`test_adjacent_points_differ_by_one_in_morton_code`、
`test_query_box_matches_brute_force_on_hundred_points`、
`test_merge_leaves_preserves_query_results`、
`test_octree_batch_matches_separate_builds`。

## 约束

- 只允许标准库（`math` 或不用）+ pytest；不引入第三方包，不联网。
- 无随机数、不读时间/环境变量：同一输入必须永远得到同一棵树。
- 8 个函数的签名与返回形状固定，`octree/__init__.py` 的导出不要改。
- 所有函数就地修改并返回根节点（`insert` / `merge_leaves`），或返回新结构
  （`query_box` 返回列表、`octree_batch` 返回 `{"roots": [...]}`）。
