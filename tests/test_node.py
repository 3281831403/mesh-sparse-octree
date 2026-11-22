"""octree.node 的验收规格（15 个测试，初始全红）。

这些测试就是题目本身，不可修改：每条测试都直接调用 octree/node.py 里的桩函数，
当前 8 个函数都只有 ``raise NotImplementedError``，所以 15 条全部 failed，
且没有 collection / import / syntax error。

测试分两组：
* 前 10 条逐一钉死 10 个空间索引一致性缺陷（编码位序、掩码位数、中心面子索引、
  逐分支深度、范围查询半开区间、内部节点的值、合并条件、子节点越界、批量构建
  复用根节点、同点重复插入）；
* 后 5 条是组合/边界约束（往返、局部性、暴力对照、合并前后查询等价、
  批量与单独构建等价）。

约定：
* 只用标准库 + pytest，不使用随机数、不联网、不读时间；
* 节点一律按 node.py 模块 docstring 里的 dict 形状构造，不依赖任何隐藏状态。
"""

import pytest

from octree import (
    child_index,
    insert,
    merge_leaves,
    morton_decode,
    morton_encode,
    octree_batch,
    query_box,
    validate_tree,
)

# --------------------------------------------------------------------------
# 构造与检查用的小工具（不调用被测函数，只拼 dict / 遍历 dict）
# --------------------------------------------------------------------------


def _root(max_depth=4):
    """按契约构造一棵空的根节点。"""
    return {
        "depth": 0,
        "x": 0,
        "y": 0,
        "z": 0,
        "max_depth": max_depth,
        "points": {},
        "children": {},
    }


def _node(depth, x, y, z, points=None, children=None):
    """按契约构造一个非根节点。"""
    return {
        "depth": depth,
        "x": x,
        "y": y,
        "z": z,
        "points": dict(points or {}),
        "children": dict(children or {}),
    }


def _bits(index):
    return index & 1, (index >> 1) & 1, (index >> 2) & 1


def _center(depth, index):
    """深度 depth 下、子索引 index 对应格子的中心点。"""
    size = 1.0 / (1 << depth)
    bx, by, bz = _bits(index)
    return (bx * size + size / 2, by * size + size / 2, bz * size + size / 2)


def _count_nodes(node):
    return 1 + sum(_count_nodes(child) for child in node["children"].values())


def _stored(node):
    """收集树上所有 (depth, point, value)，先序遍历。"""
    out = [(node["depth"], point, value) for point, value in node["points"].items()]
    for child in node["children"].values():
        out.extend(_stored(child))
    return out


def _all_points_sorted(root):
    return [point for point, _value in query_box(root, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))]


def _assert_every_leaf_at(node, depth):
    """断言：内部节点的子节点坐标/深度连续，且所有叶子恰好落在 depth 层。"""
    if node["children"]:
        assert node["depth"] < depth
        for index, child in node["children"].items():
            bx, by, bz = _bits(index)
            assert child["depth"] == node["depth"] + 1
            assert (child["x"], child["y"], child["z"]) == (
                2 * node["x"] + bx,
                2 * node["y"] + by,
                2 * node["z"] + bz,
            )
            _assert_every_leaf_at(child, depth)
    else:
        assert node["depth"] == depth, "叶子深度参差：depth=%d，期望 %d" % (node["depth"], depth)


# --------------------------------------------------------------------------
# 缺陷 1：morton_encode 的位交错把 x, y, z 顺序写反，相邻点编码不连续
# --------------------------------------------------------------------------


def test_morton_encode_interleaves_x_at_low_bit():
    # 深度 2：x 占三元组最低位，y 次之，z 最高
    assert morton_encode(0.5, 0.0, 0.0, 2) == 0b001000
    assert morton_encode(0.0, 0.5, 0.0, 2) == 0b010000
    assert morton_encode(0.0, 0.0, 0.5, 2) == 0b100000
    assert morton_encode(0.5, 0.5, 0.5, 2) == 0b111000
    # 原点恒为 0；深度 0 只有一格
    assert morton_encode(0.0, 0.0, 0.0, 2) == 0
    assert morton_encode(0.0, 0.0, 0.0, 0) == 0
    assert morton_encode(0.75, 0.75, 0.75, 0) == 0
    # 深度 3 下 x 的第 3 位落到 bit 6，说明每个三元组整体左移 3*i
    assert morton_encode(0.5, 0.0, 0.0, 3) == 1 << 6


# --------------------------------------------------------------------------
# 缺陷 2：morton_decode 的位掩码少一位，最大坐标解错（往返对不上）
# --------------------------------------------------------------------------


def test_morton_decode_keeps_top_bit_for_largest_cell():
    depth = 5
    top = 31.0 / 32.0
    code = morton_encode(top, top, top, depth)
    assert code == (1 << (3 * depth)) - 1  # 三轴全 1 就是 8**depth - 1
    assert morton_decode(code, depth) == pytest.approx((top, top, top), abs=1e-9)
    # 缺了最高位时这里会解成 15/32
    assert morton_decode(code, depth)[0] > 0.9
    assert morton_decode(0, depth) == (0.0, 0.0, 0.0)
    assert morton_decode(0, 0) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# 缺陷 3：child_index 用 < 而不是 <=，落在中心面上的点归错子节点
# --------------------------------------------------------------------------


def test_child_index_puts_center_face_point_in_high_half():
    c = (0.5, 0.5, 0.5)
    # 恰好压在中心面上：三轴都算高半区
    assert child_index(0.5, 0.5, 0.5, *c) == 7
    assert child_index(0.25, 0.75, 0.5, *c) == 6
    assert child_index(0.75, 0.25, 0.5, *c) == 5
    assert child_index(0.75, 0.75, 0.25, *c) == 3
    assert child_index(0.0, 0.0, 0.0, *c) == 0
    # 数组位权：x=1、y=2、z=4
    assert child_index(0.75, 0.0, 0.0, *c) == 1
    assert child_index(0.0, 0.75, 0.0, *c) == 2
    assert child_index(0.0, 0.0, 0.75, *c) == 4
    # 中心面在任意节点上都成立：深度 1 的 (0,0,0) 节点中心是 0.25
    assert child_index(0.25, 0.1, 0.1, 0.25, 0.25, 0.25) == 1
    assert child_index(0.24, 0.24, 0.24, 0.25, 0.25, 0.25) == 0


# --------------------------------------------------------------------------
# 缺陷 4：insert 的细分深度用全局计数，不同分支深度不一致
# --------------------------------------------------------------------------


def test_insert_reaches_max_depth_on_every_branch():
    root = _root(max_depth=3)
    points = [
        (0.125, 0.125, 0.125),
        (0.875, 0.875, 0.875),
        (0.125, 0.875, 0.375),
        (0.625, 0.125, 0.875),
        (0.5, 0.5, 0.5),
    ]
    for index, point in enumerate(points):
        insert(root, point, index)

    stored = _stored(root)
    assert len(stored) == len(points)
    assert sorted(point for _depth, point, _value in stored) == sorted(points)
    # 所有点都停在 depth == max_depth，没有分支提前收口
    assert {depth for depth, _point, _value in stored} == {3}
    _assert_every_leaf_at(root, 3)
    assert validate_tree(root, 3) == []


# --------------------------------------------------------------------------
# 缺陷 5：query_box 用闭区间，边界点被重复返回 / 不应命中的点被命中
# --------------------------------------------------------------------------


def test_query_box_is_half_open_on_upper_boundary():
    root = _root(max_depth=2)
    inner = (0.25, 0.25, 0.25)
    lower = (0.5, 0.5, 0.5)
    upper = (0.75, 0.75, 0.75)
    edge = (0.7, 0.7, 0.7)
    for point in (inner, lower, upper):
        insert(root, point, point[0])

    # 下边界含、上边界不含
    assert query_box(root, ((0.5, 0.5, 0.5), (0.75, 0.75, 0.75))) == [(lower, lower[0])]
    assert query_box(root, ((0.25, 0.25, 0.25), (0.5, 0.5, 0.5))) == [(inner, inner[0])]
    # 盒子正好压在 0.5 上时，0.5 只能算给左边那个盒子
    assert [p for p, _v in query_box(root, ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))] == [inner]

    # 上界落在格子内部（0.7 在 depth 2 的格子 [0.5, 0.75) 里）时，
    # 压在上界的点同样不能命中；闭区间实现会把它带出来
    insert(root, edge, "edge")
    assert [p for p, _v in query_box(root, ((0.25, 0.25, 0.25), (0.7, 0.7, 0.7)))] == [
        inner,
        lower,
    ]
    assert [p for p, _v in query_box(root, ((0.7, 0.7, 0.7), (0.9, 0.9, 0.9)))] == [
        edge,
        upper,
    ]

    # 整个立方体（上界 1.0 半开）里每个点各返回一次，不重复
    hits = query_box(root, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    assert [p for p, _v in hits] == [inner, lower, edge, upper]
    assert len(hits) == 4


# --------------------------------------------------------------------------
# 缺陷 6：query_box 只查叶节点，挂在内部节点上的值丢失
# --------------------------------------------------------------------------


def test_query_box_reads_values_stored_on_internal_nodes():
    # (0.25,0.25,0.25)：depth 1 节点； (0.375,...)：depth 2 叶子； (0,0,0)：根节点
    mid_point = (0.25, 0.25, 0.25)
    leaf_point = (0.375, 0.375, 0.375)
    root_point = (0.0, 0.0, 0.0)
    grandchild = _node(2, 1, 1, 1, points={leaf_point: "leaf"})
    middle = _node(1, 0, 0, 0, points={mid_point: "middle"}, children={7: grandchild})
    root = _root(max_depth=2)
    root["points"] = {root_point: "root"}
    root["children"] = {0: middle}

    assert validate_tree(root, 2) == []
    hits = query_box(root, ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    assert hits == [
        (root_point, "root"),
        (mid_point, "middle"),
        (leaf_point, "leaf"),
    ]
    # 剪枝盒子只覆盖根节点的一角：根节点上的点仍在盒子里
    assert query_box(root, ((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))) == [(root_point, "root")]


# --------------------------------------------------------------------------
# 缺陷 7：merge_leaves 合并时只比较值不比较子节点数，误合并
# --------------------------------------------------------------------------


def test_merge_leaves_requires_eight_leaf_siblings_with_equal_values():
    # (a) 8 个兄弟叶子、值相同：允许合并
    full = _root()
    for index in range(8):
        full["children"][index] = _node(1, *_bits(index), points={_center(1, index): "A"})
    merged = merge_leaves(full)
    assert merged["children"] == {}
    assert len(merged["points"]) == 8
    assert _count_nodes(merged) == 1

    # (b) 只有 3 个兄弟：子节点数不齐，不许把父节点压成叶子
    sparse = _root()
    for index in (0, 1, 2):
        sparse["children"][index] = _node(1, *_bits(index), points={_center(1, index): "A"})
    out = merge_leaves(sparse)
    assert sorted(out["children"]) == [0, 1, 2]
    assert out["points"] == {}
    assert _count_nodes(out) == 4

    # (c) 8 个兄弟但值不同：不许合并
    mixed = _root()
    for index in range(8):
        value = "B" if index == 5 else "A"
        mixed["children"][index] = _node(1, *_bits(index), points={_center(1, index): value})
    out = merge_leaves(mixed)
    assert sorted(out["children"]) == list(range(8))
    assert out["points"] == {}

    # (d) 兄弟里有非叶节点：不许只把其它 7 个叶子抽上来合并
    nested = _root()
    for index in range(8):
        if index == 3:
            grandchild = _node(2, 2, 2, 0, points={(0.625, 0.625, 0.125): "A"})
            nested["children"][index] = _node(
                1, *_bits(index), children={0: grandchild}
            )
        else:
            nested["children"][index] = _node(1, *_bits(index), points={_center(1, index): "A"})
    out = merge_leaves(nested)
    assert sorted(out["children"]) == list(range(8))
    assert out["points"] == {}
    assert _count_nodes(out) == 10


# --------------------------------------------------------------------------
# 缺陷 8：validate_tree 不检查子节点是否越父节点边界，越界未报
# --------------------------------------------------------------------------


def test_validate_tree_reports_child_outside_parent_cell():
    good = _root()
    for index in range(8):
        good["children"][index] = _node(1, *_bits(index), points={_center(1, index): index})
    assert validate_tree(good, 1) == []

    bad = _root()
    for index in range(8):
        bad["children"][index] = _node(1, *_bits(index), points={_center(1, index): index})
    # 7 号子节点覆盖 [0.5,1)³，却在下面挂了一个覆盖 [0,0.25)³ 的孙节点：
    # 子节点的格子跑到父格子外面，必须报出来
    bad["children"][7]["children"][0] = _node(2, 0, 0, 0, points={(0.125, 0.125, 0.125): "outside"})
    errors = validate_tree(bad, 2)
    assert errors, "子节点越出父节点边界却没有报错"
    assert all(isinstance(message, str) and message for message in errors)

    # 空叶同样要报：depth 1 的节点既没有点也没有子节点
    empty_leaf = _root()
    empty_leaf["children"][0] = _node(1, 0, 0, 0)
    assert validate_tree(empty_leaf, 2) != []


# --------------------------------------------------------------------------
# 缺陷 9：octree_batch 复用根节点，第二批点集挂到第一批树上
# --------------------------------------------------------------------------


def test_octree_batch_builds_independent_trees_per_batch():
    batch_one = {(0.125, 0.125, 0.125): "a1", (0.625, 0.625, 0.625): "a2"}
    batch_two = {(0.875, 0.125, 0.375): "b1"}
    roots = octree_batch([batch_one, batch_two], {"max_depth": 2})["roots"]

    assert len(roots) == 2
    assert roots[0] is not roots[1], "两批点集复用了同一个根节点"
    assert roots[0]["points"] == {}
    assert roots[1]["points"] == {}
    assert _all_points_sorted(roots[0]) == sorted(batch_one)
    assert _all_points_sorted(roots[1]) == sorted(batch_two)

    # 往第一棵树里插点，第二棵树必须不受影响
    insert(roots[0], (0.5, 0.5, 0.5), "late")
    assert len(_all_points_sorted(roots[0])) == 3
    assert query_box(roots[1], ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) == [
        (point, batch_two[point]) for point in sorted(batch_two)
    ]


# --------------------------------------------------------------------------
# 缺陷 10：insert 对同一点重复插入不更新值，后写失败
# --------------------------------------------------------------------------


def test_insert_duplicate_point_overwrites_value():
    root = _root(max_depth=2)
    point = (0.375, 0.375, 0.375)
    insert(root, point, "first")
    nodes_before = _count_nodes(root)
    insert(root, point, "second")

    assert _count_nodes(root) == nodes_before, "重复插入同一点不应新增节点"
    hits = query_box(root, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    assert hits == [(point, "second")]
    # 树上只保留一份
    stored = [item for item in _stored(root) if item[1] == point]
    assert len(stored) == 1 and stored[0][2] == "second"


# --------------------------------------------------------------------------
# 组合/边界 1：encode/decode 往返精确（容差 1e-9）
# --------------------------------------------------------------------------


def test_morton_encode_decode_round_trip_within_tolerance():
    for depth in range(1, 6):
        step = 1.0 / (1 << depth)
        largest = ((1 << depth) - 1) * step
        samples = [
            (0.0, 0.0, 0.0),
            (largest, largest, largest),
            (0.5, 0.0, 0.0),
            (step, largest, 3 * step if 3 * step < 1.0 else step),
        ]
        for point in samples:
            decoded = morton_decode(morton_encode(*point, depth), depth)
            assert isinstance(decoded, tuple) and len(decoded) == 3
            assert decoded == pytest.approx(point, abs=1e-9)
            # 返回值是格子原点，落在 [0,1)
            assert all(0.0 <= value < 1.0 for value in decoded)


# --------------------------------------------------------------------------
# 组合/边界 2：相邻点 Morton 码差 1（局部性）
# --------------------------------------------------------------------------


def test_adjacent_points_differ_by_one_in_morton_code():
    depth = 4
    step = 1.0 / 16.0
    codes = [morton_encode(index * step, 0.125, 0.25, depth) for index in range(16)]

    # 沿 x 走，编码必须单调递增
    assert codes == sorted(codes)
    # z-order 曲线上，同一个 x 相邻格里偶数号格子到下一个格子恰好 +1
    for index in range(0, 16, 2):
        assert codes[index + 1] - codes[index] == 1, (
            "相邻格子的编码差了 %d，位交错顺序写反了" % (codes[index + 1] - codes[index])
        )
    # y、z 方向的位权分别是 2 和 4（x 在最低位）
    base = morton_encode(0.0, 0.0, 0.0, depth)
    assert morton_encode(step, 0.0, 0.0, depth) - base == 1
    assert morton_encode(0.0, step, 0.0, depth) - base == 2
    assert morton_encode(0.0, 0.0, step, depth) - base == 4


# --------------------------------------------------------------------------
# 组合/边界 3：100 个点的查询结果与暴力扫描一致
# --------------------------------------------------------------------------


def test_query_box_matches_brute_force_on_hundred_points():
    root = _root(max_depth=4)
    points = {}
    for index in range(100):
        point = (
            ((index * 37) % 101) / 101.0,
            ((index * 59) % 101) / 101.0,
            ((index * 83) % 101) / 101.0,
        )
        points[point] = index
        insert(root, point, index)
    assert len(_stored(root)) == 100

    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.25, 0.5, 0.1), (0.75, 0.9, 0.6)),
        ((0.4, 0.4, 0.4), (0.41, 0.41, 0.41)),
        ((0.9, 0.9, 0.9), (1.0, 1.0, 1.0)),
    ]
    for box in boxes:
        low, high = box
        expected = sorted(
            (point, value)
            for point, value in points.items()
            if all(low[axis] <= point[axis] < high[axis] for axis in range(3))
        )
        assert query_box(root, box) == expected
    assert len(query_box(root, boxes[0])) == 100


# --------------------------------------------------------------------------
# 组合/边界 4：合并后查询结果不变
# --------------------------------------------------------------------------


def test_merge_leaves_preserves_query_results():
    root = _root(max_depth=2)
    points = {}
    for gx in range(4):
        for gy in range(4):
            for gz in range(4):
                point = ((gx + 0.5) / 4.0, (gy + 0.5) / 4.0, (gz + 0.5) / 4.0)
                points[point] = "same"
                insert(root, point, "same")

    boxes = [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)),
        ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ]
    before = [query_box(root, box) for box in boxes]
    nodes_before = _count_nodes(root)
    merge_leaves(root)
    nodes_after = _count_nodes(root)

    assert nodes_after < nodes_before, "8 个同值兄弟叶应当被合并"
    assert [query_box(root, box) for box in boxes] == before
    assert query_box(root, boxes[0]) == sorted(points.items())
    assert validate_tree(root, 2) == []


# --------------------------------------------------------------------------
# 组合/边界 5：两批独立构建与单独构建相同
# --------------------------------------------------------------------------


def test_octree_batch_matches_separate_builds():
    batch_one = {((index + 0.5) / 8.0, 0.125, 0.375): index for index in range(4)}
    batch_two = {(0.625, (index + 0.5) / 8.0, 0.875): -index for index in range(3)}
    config = {"max_depth": 3}

    both = octree_batch([batch_one, batch_two], config)["roots"]
    only_one = octree_batch([batch_one], config)["roots"][0]
    only_two = octree_batch([batch_two], config)["roots"][0]

    assert len(both) == 2
    assert both[0] == only_one
    assert both[1] == only_two
    assert both[0] != both[1]
    assert both[0]["depth"] == 0 and both[1]["depth"] == 0
    assert both[0] is not only_one
    assert query_box(both[0], ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) == sorted(
        batch_one.items()
    )
    assert query_box(both[1], ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) == sorted(
        batch_two.items()
    )
    assert octree_batch([], config) == {"roots": []}
