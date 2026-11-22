"""稀疏八叉树的节点编码与空间索引原语（8 个桩函数）。

坐标与网格
----------
所有坐标都定义在单位立方体 ``[0.0, 1.0)³`` 上，区间**左闭右开**。
深度 ``d`` 的网格把立方体切成 ``2**d × 2**d × 2**d`` 个格子，格子边长
``2**-d``，用整数三元组 ``(x, y, z)`` 标识，且 ``0 <= x, y, z < 2**d``。
格子 ``(x, y, z)`` 在深度 ``d`` 覆盖的半开立方体是::

    [x / 2**d, (x + 1) / 2**d) × [y / 2**d, (y + 1) / 2**d) × [z / 2**d, (z + 1) / 2**d)

深度上限为 10（``max_depth`` 取 0..10 的整数），即最细一层格子边长为 2**-10。

节点字典
--------
整棵树由普通 dict 表示，形状固定，测试会直接按这个形状构造节点::

    {
        "depth": 0,          # int，根为 0，子节点 = 父节点 depth + 1
        "x": 0,              # int，该深度网格下的格子坐标，0 <= x, y, z < 2**depth
        "y": 0,
        "z": 0,
        "max_depth": 6,      # 仅根节点必须携带；插入时的深度上限
        "points": {},        # {point: value}，point 是三元 float 组
        "children": {},      # {child_index: node}，child_index 取 0..7
    }

子节点索引按位定义：``bit0 = x``、``bit1 = y``、``bit2 = z``，即子节点索引
``i`` 对应的格子坐标是 ``(2*px + (i & 1), 2*py + ((i >> 1) & 1), 2*pz + ((i >> 2) & 1))``。
点可以挂在叶节点上，也可以挂在内部节点上（合并、外部构造都会产生这种树），
查询与校验都必须按“节点自己 points 里的点”来算，不能只看叶子。

浮点约定
--------
格子的原点、边长都是 2 的幂，属于可精确表示的浮点数；编码往返、范围查询等
涉及浮点比较的地方一律允许 1e-9 的绝对误差（``abs=1e-9``）。

初始状态：下面 8 个函数全部只有 ``raise NotImplementedError``，
``tests/test_node.py`` 的 15 个测试因此全红。
"""


def morton_encode(x, y, z, depth):
    """把单位立方体内的点编码成 Morton 码（Z-order，位交错）。

    先按深度量化到网格：``ix = min(int(x * 2**depth), 2**depth - 1)``
    （``ix = max(ix, 0)``；y、z 同理），再把 ``ix / iy / iz`` 的二进制位
    按三元组交错，**x 占每个三元组的最低位**：

        对 i in range(depth)：
            code |= ((ix >> i) & 1) << (3 * i)
            code |= ((iy >> i) & 1) << (3 * i + 1)
            code |= ((iz >> i) & 1) << (3 * i + 2)

    因此返回值落在 ``[0, 8**depth)``：``depth == 0`` 恒返回 0；深度 2 时
    ``(0.5, 0, 0) -> 0b001000``、``(0, 0.5, 0) -> 0b010000``、
    ``(0, 0, 0.5) -> 0b100000``；同一 y/z 上 ``x`` 相邻的两个格子，
    编码在 z-order 曲线上相邻（偶数号格子到下一个格子恰好 +1，见
    ``morton_decode`` 与测试里的局部性断言）。

    参数 depth 取 0..10 的整数。返回 int，不做任何隐式取整以外的变换。
    """
    n = 1 << depth
    ix = max(min(int(x * n), n - 1), 0)
    iy = max(min(int(y * n), n - 1), 0)
    iz = max(min(int(z * n), n - 1), 0)

    code = 0
    for i in range(depth):
        code |= ((ix >> i) & 1) << (3 * i)
        code |= ((iy >> i) & 1) << (3 * i + 1)
        code |= ((iz >> i) & 1) << (3 * i + 2)
    return code


def morton_decode(code, depth):
    """``morton_encode`` 的逆运算：反解出格子坐标（归一化到 [0, 1) 的浮点原点）。

    按 ``morton_encode`` 相同的位序取出 ``ix / iy / iz``，返回格子**下界**：

        (ix / 2**depth, iy / 2**depth, iz / 2**depth)

    即返回值 ``p`` 满足“点 p 落在编码为 code 的那个格子内”，而不是格子中心。
    对任意格子对齐的点 ``p``（每个分量都是 ``k / 2**depth`` 的形式）都有::

        morton_decode(morton_encode(p, depth), depth) == p   # 绝对误差 <= 1e-9

    ``depth == 0`` 时 code 必须是 0，返回 ``(0.0, 0.0, 0.0)``。
    实现时位掩码必须覆盖全部 ``3 * depth`` 位：深度 d 下最大的格子
    （``x = y = z = 2**d - 1``）编码为 ``8**d - 1``，反解必须还原它，
    掩码少一位会把最高位吃掉、最大坐标解成一半。
    返回三元 float 组。
    """
    ix = iy = iz = 0
    for i in range(depth):
        ix |= ((code >> (3 * i)) & 1) << i
        iy |= ((code >> (3 * i + 1)) & 1) << i
        iz |= ((code >> (3 * i + 2)) & 1) << i

    n = 1 << depth
    return (ix / n, iy / n, iz / n)


def child_index(px, py, pz, cx, cy, cz):
    """返回点 ``(px, py, pz)`` 在中心为 ``(cx, cy, cz)`` 的节点里所属子节点的索引 0..7。

    逐轴判定位，**点落在高半区（含中心面）时该轴位为 1**::

        bx = 1 if px >= cx else 0
        by = 1 if py >= cy else 0
        bz = 1 if pz >= cz else 0
        return bx | (by << 1) | (bz << 2)

    也就是说坐标恰好等于中心面的点归入高半区；判据一旦写成严格不等号
    （``px < cx`` 取反、``px > cx`` 取 1），落在中心面上的点就会被划进低半区，
    同一个点会同时落进两个相邻格子的管辖范围，树里出现重复值。
    中心面三轴都命中的点（``px == cx and py == cy and pz == cz``）返回 7。
    """
    bx = 1 if px >= cx else 0
    by = 1 if py >= cy else 0
    bz = 1 if pz >= cz else 0
    return bx | (by << 1) | (bz << 2)


def insert(root, point, value):
    """把点 ``point = (x, y, z)`` 与值 ``value`` 插入以 root 为根的树，返回根节点。

    契约：

    * ``root`` 必须是根节点字典，且带 ``max_depth``（0..10 的整数）；
    * ``point`` 必须落在 ``[0.0, 1.0)³`` 内，否则抛 ``ValueError``；
    * 从根开始按 ``child_index`` 逐层下行：先看当前节点是否已经存了**同一个**
      ``point``，是则原地覆盖 value 并立即返回（同一个点重复插入以最后一次为准，
      树上只保留一份，不会因为重复插入多长出一条分支）；否则若
      ``depth == max_depth`` 就把点存进当前节点的 ``points``，否则确保对应子节点
      存在（子节点坐标 = ``2 * 父坐标 + 该位``）再继续下行；
    * 深度上限只由 ``max_depth`` 决定，并且是**逐分支**生效的：每一条从根到叶的
      路径长度都恰好是 ``max_depth``，不能因为先插了别的点就变浅或变深，
      同一棵树里不能出现深度参差的叶子；
    * 允许在已有内部节点的树上插入：不动其它分支，也不丢弃已挂的值。

    返回根节点本身（就地修改）。空树（``points == {}`` 且 ``children == {}``）
    插入后必须成为一条到 ``max_depth`` 的路径。
    """
    px, py, pz = point
    if not all(0.0 <= value < 1.0 for value in (px, py, pz)):
        raise ValueError("point %r 不在 [0.0, 1.0)³ 内" % (point,))

    max_depth = root["max_depth"]
    node = root
    while True:
        if point in node["points"]:
            node["points"][point] = value
            return root
        if node["depth"] >= max_depth:
            node["points"][point] = value
            return root

        # 当前节点的中心（节点覆盖 [x / 2**d, (x + 1) / 2**d)）
        scale = 1 << node["depth"]
        cx = (node["x"] + 0.5) / scale
        cy = (node["y"] + 0.5) / scale
        cz = (node["z"] + 0.5) / scale
        index = child_index(px, py, pz, cx, cy, cz)
        bx, by, bz = index & 1, (index >> 1) & 1, (index >> 2) & 1

        child = node["children"].get(index)
        if child is None:
            child = {
                "depth": node["depth"] + 1,
                "x": 2 * node["x"] + bx,
                "y": 2 * node["y"] + by,
                "z": 2 * node["z"] + bz,
                "points": {},
                "children": {},
            }
            node["children"][index] = child
        node = child


def query_box(root, box):
    """范围查询：返回落在半开 AABB 内的 ``(point, value)`` 列表，按 point 升序排序。

    ``box = ((x0, y0, z0), (x1, y1, z1))``，点的判定是**左闭右开**::

        x0 <= px < x1 and y0 <= py < y1 and z0 <= pz < z1

    上边界上的点不属于该盒子，下边界上的点属于；一个点最多返回一次
    （即使树里错误地在两个节点各存了一份，也要按键去重）。
    节点自身的格子与 box 不相交时可以把整棵子树剪掉；但**不能**假设值只挂在
    叶子上——内部节点的 ``points`` 里同样可能有落在盒子里的点（合并会把值上提），
    这些点必须被返回。

    返回 ``[(point, value), ...]``，point 是三元组，按字典序升序排序；
    没有命中时返回 ``[]``。
    """
    (x0, y0, z0), (x1, y1, z1) = box
    low = (x0, y0, z0)
    high = (x1, y1, z1)
    found = {}

    def visit(node):
        depth = node["depth"]
        size = 1.0 / (1 << depth)
        cell_low = (node["x"] * size, node["y"] * size, node["z"] * size)
        cell_high = (
            (node["x"] + 1) * size,
            (node["y"] + 1) * size,
            (node["z"] + 1) * size,
        )
        # 半开区间 [cell_low, cell_high) 与 [low, high) 不相交就整棵剪掉
        for axis in range(3):
            if cell_low[axis] >= high[axis] or low[axis] >= cell_high[axis]:
                return

        for point, value in node["points"].items():
            if point in found:
                continue
            if all(low[axis] <= point[axis] < high[axis] for axis in range(3)):
                found[point] = value

        for child in node["children"].values():
            visit(child)

    visit(root)
    return sorted(found.items(), key=lambda item: item[0])


def merge_leaves(root):
    """自底向上合并「八个兄弟完全相同」的叶子，返回根节点（就地修改）。

    某个节点满足下面**全部**条件时，才允许把它的 8 个子节点压成自己：

    * 恰好有 8 个子节点，键 0..7 全在（稀疏地只有 3 个、5 个兄弟时不能合并，
      否则父节点的格子把没有数据的邻域也吞进去了）；
    * 8 个子节点都是叶子（``children`` 为空），且各自的 ``points`` 非空
      （不允许制造空叶）；
    * 8 个子节点里所有 value 都相等（用 ``==`` 逐值比较，不假设 value 可哈希）。

    满足时把这 8 个叶子的 ``points`` 全部并到父节点的 ``points``（值相同、
    点可以不同，合并后父节点持有多个点），并清空父节点的 ``children``，
    父节点成为新的叶子；任一条不满足就保持原样。合并只允许减少节点数，
    不允许改变任何点的值，所以对任意盒子和合并前一定完全相同——这条是
    ``tests/test_node.py`` 里“合并后查询结果不变”的依据。
    """
    def visit(node):
        for child in list(node["children"].values()):
            visit(child)

        children = node["children"]
        if set(children) != set(range(8)):
            return
        leaves = list(children.values())
        if any(child["children"] for child in leaves):
            return
        if any(not child["points"] for child in leaves):
            return

        first_value = next(iter(leaves[0]["points"].values()))
        for child in leaves:
            for value in child["points"].values():
                if not (value == first_value):
                    return

        merged = {}
        for child in leaves:
            merged.update(child["points"])
        node["points"].update(merged)
        node["children"] = {}

    visit(root)
    return root


def validate_tree(root, depth):
    """检查树的结构一致性，返回错误字符串列表；合法树返回 ``[]``。

    ``depth`` 是这棵树允许的最大深度。按先序遍历逐节点检查，错误信息按遍历顺序
    追加，每条都带上出错的 ``depth`` 与 ``(x, y, z)``，便于定位：

    * 根节点 ``depth`` 必须为 0，``x == y == z == 0``；
    * 每个节点的 ``depth`` 在 ``0..depth`` 之内，``x / y / z`` 在 ``[0, 2**depth)`` 内；
    * ``children`` 的键必须是 0..7 的整数；子节点 ``depth`` 必须是父节点
      ``depth + 1``，坐标必须是 ``2 * 父坐标 + 该键对应位``，也就是子节点的格子
      必须完整落在父节点格子内——**越出父节点边界的子树必须报错**，
      不能只看深度对就放行；
    * 叶节点（``children`` 为空）的 ``points`` 不能为空，禁止空叶；
    * 节点 ``points`` 里的每个点必须落在该节点格子内（半开区间）。

    只返回错误描述，不抛异常，也不修改树。合法的空树（只有根、没有点）返回 ``[]``。
    """
    EPS = 1e-9
    errors = []

    def label(node):
        return "depth=%s, (x=%s, y=%s, z=%s)" % (
            node.get("depth"),
            node.get("x"),
            node.get("y"),
            node.get("z"),
        )

    def check_node(node, parent):
        prefix = label(node)
        node_depth = node["depth"]
        depth_ok = isinstance(node_depth, int) and 0 <= node_depth <= depth

        if not depth_ok:
            errors.append("%s: 节点深度越界（允许 0..%d）" % (prefix, depth))
        if parent is None:
            if node_depth != 0 or node["x"] != 0 or node["y"] != 0 or node["z"] != 0:
                errors.append("%s: 根节点必须是 depth=0 且坐标为 (0, 0, 0)" % prefix)
        else:
            if node_depth != parent["depth"] + 1:
                errors.append(
                    "%s: 子节点深度应为父节点深度 + 1（父 %s）"
                    % (prefix, label(parent))
                )
            # 子节点坐标的具体期望值由父节点遍历子键时校验（见下）

        if depth_ok:
            limit = 1 << node_depth
            for axis in ("x", "y", "z"):
                value = node[axis]
                if not isinstance(value, int) or not (0 <= value < limit):
                    errors.append(
                        "%s: 坐标 %s 应在 [0, %d) 内" % (prefix, axis, limit)
                    )

        # 点必须落在本节点格子 [x/2**d, (x+1)/2**d) 内
        if depth_ok and node["points"]:
            size = 1.0 / (1 << node_depth)
            lower = (node["x"] * size, node["y"] * size, node["z"] * size)
            upper = (
                (node["x"] + 1) * size,
                (node["y"] + 1) * size,
                (node["z"] + 1) * size,
            )
            for point in node["points"]:
                ok = True
                for axis in range(3):
                    if not (lower[axis] - EPS <= point[axis] < upper[axis]):
                        ok = False
                        break
                if not ok:
                    errors.append(
                        "%s: 点 %r 不在节点格子内" % (prefix, point)
                    )

        children = node["children"]
        if not children:
            # 合法的空树只有根，其它任何空叶都算错误
            if not node["points"] and parent is not None:
                errors.append("%s: 空叶节点（既没有点也没有子节点）" % prefix)
            return

        for index in sorted(children, key=repr):
            child = children[index]
            if not isinstance(index, int) or not (0 <= index < 8):
                errors.append("%s: 子节点键 %r 不是 0..7 的整数" % (prefix, index))
                check_node(child, node)
                continue
            bx, by, bz = index & 1, (index >> 1) & 1, (index >> 2) & 1
            expected = (
                2 * node["x"] + bx,
                2 * node["y"] + by,
                2 * node["z"] + bz,
            )
            actual = (child["x"], child["y"], child["z"])
            if actual != expected:
                errors.append(
                    "%s: 子节点 %r 的坐标 %r 越出父节点格子，应为 %r"
                    % (prefix, index, actual, expected)
                )
            check_node(child, node)

    check_node(root, None)
    return errors


def octree_batch(points, config):
    """用同一份 config 批量构建多棵**互相独立**的八叉树。

    参数：

    * ``points``：点集列表，每个元素是 ``{point: value}`` 字典，一个元素一棵树；
      空列表返回 ``{"roots": []}``；
    * ``config``：``{"max_depth": int}``，0..10。

    返回 ``{"roots": [root, ...]}``，顺序与 ``points`` 一致。第 i 棵树只包含
    第 i 个点集的点；任意两棵树之间**不共享任何节点**（改一棵不会影响另一棵），
    并且与单独调用 ``octree_batch([第 i 个点集], config)`` 得到的树在结构上
    完全相等（``==``）。空点集产生一棵空的根节点（``points == {}``、
    ``children == {}``），不留任何分支。深度上限来自 ``config["max_depth"]``，
    并写进每棵树的根节点。
    """
    max_depth = config["max_depth"]
    roots = []
    for point_set in points:
        root = {
            "depth": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "max_depth": max_depth,
            "points": {},
            "children": {},
        }
        for point, value in point_set.items():
            insert(root, point, value)
        roots.append(root)
    return {"roots": roots}
