# L3: 結構錨點與孤島分類

## 範圍
本文件描述 `api` 模組中負責結構支撐、錨點追蹤、孤島分裂與 orphan 通知的核心流程。

## 核心類別

| 類別 | 套件 | 職責 |
| --- | --- | --- |
| `AnchorContinuityChecker` | `com.blockreality.api.physics` | 判定 `RBlock` 是否具天然錨點或顯式錨定狀態。 |
| `StructureIslandRegistry` | `com.blockreality.api.physics` | 維護結構 island、anchor 集合、split/orphan 分類與 listener 通知。 |
| `BlockPhysicsEventHandler` | `com.blockreality.api.event` | 接收 Forge 放置/破壞事件，將方塊拓撲變化同步到 registry、anchor 狀態與 PFSF。 |

## 天然錨點規則

`AnchorContinuityChecker.isNaturalAnchor(ServerLevel, BlockPos)` 目前將以下情況視為天然錨點：

1. 方塊位於 `minBuildHeight + 1` 或更低。
2. 下方是 `bedrock` 或 `barrier`。
3. 方塊本身是 `bedrock` 或 `barrier`。
4. 該位置的 `RBlockEntity` 為 `BlockType.ANCHOR_PILE`。
5. 任一六向鄰居是非流體、非空氣、非 `RBlock` 的世界方塊，且其接觸面對 `RBlock` 為 `face sturdy`。

第 5 條用來覆蓋最常見的遊戲內情境：`RBlock` 建在泥土、石頭、牆面或其他原版堅固方塊上時，也應視為有物理支撐，而不是只有碰到 bedrock 才算錨定。

## 事件驅動同步

`BlockPhysicsEventHandler` 的同步規則如下：

- 放置 `RBlock` 時：
  - 先以 `isNaturalAnchor(...)` 預判是否需要 `registerAnchor(...)`。
  - 再呼叫 `registerBlock(...)` 進入 island registry。
  - 之後照常執行 RC fusion 檢查。
- 放置非 `RBlock` 時：
  - 不進 registry。
  - 但會排程檢查六向相鄰 `RBlock` 是否因此獲得新的天然支撐。
- 破壞 `RBlock` 時：
  - 執行 `unregisterBlock(...)`、`unregisterAnchor(...)`。
  - 將受影響的 island 透過 `PFSFEngine.notifyBlockChange(...)` 標記為需重建。
- 破壞非 `RBlock` 時：
  - 不直接操作 registry 成員。
  - 但會排程檢查六向相鄰 `RBlock` 是否因此失去天然支撐。
  - 若 anchor 狀態改變，會呼叫 `StructureIslandRegistry.refreshAnchorState(...)` 立即重跑 orphan 分類。

這個流程用來處理「玩家挖掉支撐 `RBlock` 的泥土/石頭，但沒有直接打掉 `RBlock` 本體」的情況，避免 structure registry 仍保留過期 anchor，導致懸空結構不倒。

## Island 重分類

`StructureIslandRegistry` 新增兩個對外輔助方法：

- `isAnchorRegistered(BlockPos)`：查詢指定位置目前是否在 anchor 集合中。
- `refreshAnchorState(ServerLevel, int, long)`：當 island 周圍世界支撐改變，但 island 成員本身沒有增減時，強制重跑 anchor/orphan 分類。

另外，`unregisterBlock(...)` 的 fracture 路徑現在會以 `allowAnchorlessClassification=true` 重跑 `checkAndSplitIsland(...)`。這代表：

- 若同一次斷裂剛好讓最後一個 anchor 消失，orphan 仍會在同一個 fracture call 內被回報。
- `advanceTopology()` 的 safety valve 仍然保留，避免在沒有任何 anchor 註冊的被動拓撲掃描中造成全圖 orphan flood。

## 測試覆蓋

`IslandSplitAnchorTrackingTest` 目前涵蓋：

- 切斷仍然各自有 anchor 的橋梁時，不應誤報 orphan。
- 拆掉唯一支撐後，orphan listener 必須同 call 觸發。
- 多分支 split 時，每個孤立手臂都要正確分類。
- 直接 fracture 導致 anchor 清空時，仍要即時回報 orphan。
- 外部支撐消失並透過 `refreshAnchorState(...)` 重判時，整個 island 必須立刻轉為 orphan。
