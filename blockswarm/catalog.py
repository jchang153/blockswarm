"""The shop's item catalog: Minecraft building blocks, each a full cube in this world.

Items come from Minecraft's Building Blocks and Colored Blocks tabs (minecraft.wiki,
"Creative inventory"). Non-cube shapes (stairs, slabs, fences) are left out; doors and glass
panes are kept and occupy a full cell like every other block.

Properties:
- `transparent`: doesn't block line of sight or view (glass, panes, sea lantern is NOT).
- `door`: agents can walk through it (they open it), but it counts as closed for air and
  enclosure analysis.
- `color`: base colour for viewers (hex), so every item renders distinctly.
Prices are in coins and roughly follow how common the material is in Minecraft.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    family: str
    price: int
    color: str
    transparent: bool = False
    door: bool = False

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "family": self.family, "price": self.price,
                "color": self.color, "transparent": self.transparent, "door": self.door}


_WOODS = [  # (id prefix, display, planks colour, log colour)
    ("oak", "Oak", "#b8945f", "#6e5634"),
    ("spruce", "Spruce", "#7a5a34", "#3b2a17"),
    ("birch", "Birch", "#d7c185", "#d8d6cc"),
    ("jungle", "Jungle", "#b4805a", "#554219"),
    ("acacia", "Acacia", "#ad5d32", "#676157"),
    ("dark_oak", "Dark Oak", "#4c3223", "#3c2e1a"),
    ("mangrove", "Mangrove", "#773631", "#5b4a2e"),
    ("cherry", "Cherry", "#e3b3ad", "#3b1f2a"),
]

_COLORS = [  # Minecraft dye colours used for wool, concrete, terracotta and stained glass
    ("white", "White", "#e9ecec"), ("light_gray", "Light Gray", "#8e8e86"),
    ("gray", "Gray", "#3e4447"), ("black", "Black", "#141519"),
    ("brown", "Brown", "#724728"), ("red", "Red", "#a12722"),
    ("orange", "Orange", "#f07613"), ("yellow", "Yellow", "#f8c527"),
    ("lime", "Lime", "#70b919"), ("green", "Green", "#546d1b"),
    ("cyan", "Cyan", "#158991"), ("light_blue", "Light Blue", "#3aafd9"),
    ("blue", "Blue", "#35399d"), ("purple", "Purple", "#792aac"),
    ("magenta", "Magenta", "#bd44b3"), ("pink", "Pink", "#ed8dac"),
]


def _build() -> list[Item]:
    items: list[Item] = [
        Item("dirt", "Dirt", "earth", 1, "#86603c"),
        Item("cobblestone", "Cobblestone", "stone", 1, "#7f7f7f"),
        Item("mossy_cobblestone", "Mossy Cobblestone", "stone", 2, "#6e7a5a"),
        Item("stone", "Stone", "stone", 2, "#7d7d7d"),
        Item("smooth_stone", "Smooth Stone", "stone", 3, "#9e9e9e"),
        Item("stone_bricks", "Stone Bricks", "stone", 3, "#7a7a7a"),
        Item("mossy_stone_bricks", "Mossy Stone Bricks", "stone", 3, "#737a63"),
        Item("granite", "Granite", "stone", 2, "#9a6a55"),
        Item("polished_granite", "Polished Granite", "stone", 3, "#9a6b59"),
        Item("diorite", "Diorite", "stone", 2, "#bcbcbc"),
        Item("polished_diorite", "Polished Diorite", "stone", 3, "#c1c1c3"),
        Item("andesite", "Andesite", "stone", 2, "#888888"),
        Item("polished_andesite", "Polished Andesite", "stone", 3, "#848685"),
        Item("cobbled_deepslate", "Cobbled Deepslate", "deepslate", 3, "#4d4d51"),
        Item("polished_deepslate", "Polished Deepslate", "deepslate", 4, "#484849"),
        Item("deepslate_bricks", "Deepslate Bricks", "deepslate", 4, "#464646"),
        Item("deepslate_tiles", "Deepslate Tiles", "deepslate", 4, "#363637"),
        Item("tuff_bricks", "Tuff Bricks", "tuff", 4, "#62675f"),
        Item("bricks", "Bricks", "bricks", 4, "#966253"),
        Item("mud_bricks", "Mud Bricks", "bricks", 3, "#89684f"),
        Item("packed_mud", "Packed Mud", "bricks", 2, "#8e6b50"),
        Item("sandstone", "Sandstone", "sandstone", 2, "#d8cb9b"),
        Item("cut_sandstone", "Cut Sandstone", "sandstone", 3, "#d9cd9e"),
        Item("smooth_sandstone", "Smooth Sandstone", "sandstone", 3, "#e0d6aa"),
        Item("red_sandstone", "Red Sandstone", "sandstone", 3, "#b5621f"),
        Item("quartz_block", "Block of Quartz", "quartz", 6, "#ece6df"),
        Item("quartz_bricks", "Quartz Bricks", "quartz", 6, "#eae5dd"),
        Item("nether_bricks", "Nether Bricks", "nether", 4, "#2c1519"),
        Item("red_nether_bricks", "Red Nether Bricks", "nether", 5, "#460709"),
        Item("blackstone", "Blackstone", "nether", 3, "#2a2328"),
        Item("polished_blackstone_bricks", "Polished Blackstone Bricks", "nether", 5, "#302a30"),
        Item("end_stone_bricks", "End Stone Bricks", "end", 5, "#dae0a2"),
        Item("purpur_block", "Purpur Block", "end", 6, "#a97ea9"),
        Item("prismarine", "Prismarine", "ocean", 6, "#63a497"),
        Item("prismarine_bricks", "Prismarine Bricks", "ocean", 6, "#63ab9e"),
        Item("dark_prismarine", "Dark Prismarine", "ocean", 7, "#335b4b"),
        Item("sea_lantern", "Sea Lantern", "ocean", 10, "#acc8be"),
        Item("copper_block", "Block of Copper", "copper", 8, "#c06c50"),
        Item("cut_copper", "Cut Copper", "copper", 8, "#bf6b51"),
        Item("iron_block", "Block of Iron", "metal", 20, "#dcdcdc"),
        Item("gold_block", "Block of Gold", "metal", 50, "#f6d03d"),
        Item("emerald_block", "Block of Emerald", "metal", 80, "#2acb57"),
        Item("diamond_block", "Block of Diamond", "metal", 150, "#62ede4"),
        Item("hay_block", "Hay Bale", "misc", 2, "#a68a0e"),
        Item("bookshelf", "Bookshelf", "misc", 6, "#6b5236"),
        Item("glass", "Glass", "glass", 3, "#c0dce4", transparent=True),
        Item("tinted_glass", "Tinted Glass", "glass", 5, "#2c2630"),  # opaque to sight in Minecraft
        Item("glass_pane", "Glass Pane", "glass", 2, "#c0dce4", transparent=True),
        Item("oak_door", "Oak Door", "door", 6, "#8c6c3c", door=True),
        Item("spruce_door", "Spruce Door", "door", 6, "#6a4f2e", door=True),
        Item("iron_door", "Iron Door", "door", 12, "#c6c6c6", door=True),
    ]
    for wid, wname, plank, log in _WOODS:
        items.append(Item(f"{wid}_planks", f"{wname} Planks", "wood", 2, plank))
        items.append(Item(f"{wid}_log", f"{wname} Log", "wood", 3, log))
    for cid, cname, col in _COLORS:
        items.append(Item(f"{cid}_wool", f"{cname} Wool", "wool", 2, col))
        items.append(Item(f"{cid}_concrete", f"{cname} Concrete", "concrete", 3, col))
        items.append(Item(f"{cid}_terracotta", f"{cname} Terracotta", "terracotta", 3, col))
        items.append(Item(f"{cid}_stained_glass", f"{cname} Stained Glass", "glass", 4, col, transparent=True))
    return items


ITEMS: tuple[Item, ...] = tuple(_build())
CATALOG: dict[str, Item] = {i.id: i for i in ITEMS}
ITEM_IDS: tuple[str, ...] = tuple(CATALOG)

assert len(CATALOG) == len(ITEMS), "duplicate item id"


def catalog_json() -> list[dict]:
    """The catalog as logged in the episode header (`shop.catalog`)."""
    return [i.to_json() for i in ITEMS]
