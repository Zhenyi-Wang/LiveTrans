// 观众报错反馈: 转发到 n8n webhook(tellme 同款通道), 内存 Map 按 IP 限流防滥用
// 限流窗口 30s: 教堂现场共享出口 IP, 60s 会让第二位反馈者白等; 30s 仍足以防脚本轰炸
const RATE_LIMIT_MS = 30_000;

// tellme 脚本同款 webhook, 公网可达; 可用 NUXT_TELLME_WEBHOOK 运行时覆盖
const DEFAULT_TELLME_WEBHOOK =
  "https://n8n.346751.xyz/webhook/51d38d7b-bc2c-4b81-81cc-158eb5e75c47";

const lastReportAt = new Map<string, number>(); // ip -> 上次成功上报时间戳

// 浏览器跨站校验白名单: 生产走腾讯云CDN回源+Caddy, 容器收到的 Host 头不等于访问域名,
// 同源比对(origin.host === host)在该拓扑下必然误杀, 故改为显式域名白名单;
// curl 等无 Origin 的非浏览器场景放行, 由 IP 限流兜底
const ALLOWED_ORIGINS = [
  "https://realtime.hainingchurch.cn",
  "http://localhost:3000",
  "http://localhost:8081",
];

export default defineEventHandler(async (event) => {
  if (getMethod(event) !== "POST") {
    setResponseHeader(event, "Allow", "POST");
    throw createError({ statusCode: 405, statusMessage: "Method Not Allowed" });
  }
  const origin = getRequestHeader(event, "origin");
  if (origin && !ALLOWED_ORIGINS.includes(origin)) {
    console.log(`[report] 拒绝跨站 origin=${origin}`);
    throw createError({ statusCode: 403, statusMessage: "Cross-Origin Not Allowed" });
  }
  // 限流键: 优先 XFF 链尾(CDN/反代把对端 IP 追加在链尾, 区分度远好于全员共享的
  // 回源 socket IP); 直连无 XFF 时(局域网直连 8081)退回 socket 地址
  const xff = getRequestHeader(event, "x-forwarded-for");
  const xffTail = xff ? xff.split(",").pop()!.trim() : "";
  const ip = xffTail || getRequestIP(event) || "unknown";
  const now = Date.now();

  if (now - (lastReportAt.get(ip) || 0) < RATE_LIMIT_MS) {
    throw createError({ statusCode: 429, statusMessage: "Too Many Requests" });
  }
  // 先同步占位再 await: check-then-act 之间不能留 await 隙, 否则并发请求可全部穿过检查
  // 失败回滚到 prev(不能无条件 delete, 会误清同 IP 此前成功留下的锁)
  const prev = lastReportAt.get(ip);
  lastReportAt.set(ip, now);
  const rollback = () => {
    if (prev === undefined) {
      lastReportAt.delete(ip);
    } else {
      lastReportAt.set(ip, prev);
    }
  };

  // readBody 也纳入回滚覆盖: 非法 JSON 抛 400 时占位必须还原(失败不占额度)
  let body;
  try {
    body = await readBody(event);
  } catch (err) {
    rollback();
    throw err;
  }
  // 留空可提交(发占位文案); 截断防超长滥用
  const userText = String(body?.message || "").trim().slice(0, 500);
  const time = new Date().toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false });

  try {
    const config = useRuntimeConfig(event);
    const resp = await fetch(config.tellmeWebhook || DEFAULT_TELLME_WEBHOOK, {
      method: "POST",
      signal: AbortSignal.timeout(10000), // 防 n8n 挂死占住 handler
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: `[livetrans 观众反馈] ${time}\n${userText || "(no description)"}`,
      }),
    });
    if (!resp.ok) throw new Error(`webhook ${resp.status}`);
  } catch (err) {
    rollback();
    console.error(`[report] 转发失败 ip=${ip}: ${String(err).replace(/[\r\n\t]/g, " ")}`);
    // 失败不占限流额度, 用户可立刻重试
    throw createError({ statusCode: 502, statusMessage: "Failed to forward report" });
  }

  console.log(
    `[report] 已转发 ip=${ip} socket=${getRequestIP(event)} xff=${xff || "-"}: ` +
      `${(userText || "(no description)").replace(/[\r\n\t]/g, " ").slice(0, 100)}`
  );
  // Map 防无限增长: 超过 1000 条时清掉已过窗口的旧条目
  if (lastReportAt.size > 1000) {
    for (const [k, t] of lastReportAt) {
      if (now - t >= RATE_LIMIT_MS) lastReportAt.delete(k);
    }
  }
  return { ok: true };
});
