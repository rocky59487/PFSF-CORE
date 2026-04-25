package com.blockreality.api.network;

import net.minecraft.network.FriendlyByteBuf;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.fml.DistExecutor;
import net.minecraftforge.network.NetworkEvent;

import java.util.function.Supplier;

/**
 * S→C: 同步 PFSF 鎖定狀態到客戶端。
 *
 * <p>當 server 端 {@link com.blockreality.api.physics.pfsf.PFSFLockdown} 進入鎖定狀態，
 * 此封包送至所有玩家以觸發紅字 HUD 橫幅顯示。</p>
 *
 * <p>封包格式：</p>
 * <pre>
 *   [bool: locked]
 *   [string: reason]   // 僅當 locked=true 時有效，最大 256 char
 * </pre>
 */
public final class PFSFLockdownPacket {

    private static final int MAX_REASON_LENGTH = 256;

    private final boolean locked;
    private final String reason;

    public PFSFLockdownPacket(boolean locked, String reason) {
        this.locked = locked;
        this.reason = reason != null ? reason : "";
    }

    public boolean isLocked() { return locked; }
    public String getReason() { return reason; }

    public static void encode(PFSFLockdownPacket packet, FriendlyByteBuf buf) {
        buf.writeBoolean(packet.locked);
        if (packet.locked) {
            String r = packet.reason;
            if (r.length() > MAX_REASON_LENGTH) {
                r = r.substring(0, MAX_REASON_LENGTH);
            }
            buf.writeUtf(r, MAX_REASON_LENGTH);
        }
    }

    public static PFSFLockdownPacket decode(FriendlyByteBuf buf) {
        boolean locked = buf.readBoolean();
        String reason = locked ? buf.readUtf(MAX_REASON_LENGTH) : "";
        return new PFSFLockdownPacket(locked, reason);
    }

    public static void handle(PFSFLockdownPacket packet, Supplier<NetworkEvent.Context> ctx) {
        ctx.get().enqueueWork(() -> {
            DistExecutor.unsafeRunWhenOn(Dist.CLIENT, () -> () -> {
                com.blockreality.api.client.PFSFLockdownHud.setLocked(
                        packet.isLocked(), packet.getReason());
            });
        });
        ctx.get().setPacketHandled(true);
    }
}
