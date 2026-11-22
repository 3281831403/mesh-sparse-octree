"""pytest 根 conftest。

只做一件事：把仓库根目录放进 sys.path，保证在仓库根执行
`python -m pytest tests -q` 时能 `import octree`。
这里不定义任何 fixture —— fixture 里抛异常会被 pytest 记成 error，
而本题要求初始状态恰好是 15 个 failed、没有 error。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
