// 观众报错反馈: 转发到 n8n webhook(tellme 同款通道), 内存 Map 按 IP 限流防滥用
// 限流窗口 30s: 教堂现场共享出口 IP, 60s 会让第二位反馈者白等; 30s 仍足以防脚本轰炸
const RATE_LIMIT_MS = 30_000;

// tellme 脚本同款 webhook, 公网可达; 可用 NUXT_TELLME_WEBHOOK 运行时覆盖
const DEFAULT_TELLME_WEBHOOK =
  "https://n8n.346751.xyz/webhook/51d38d7b-bc2c-4b81-81cc-158eb5e75c47";

const lastReportAt = new Map<string, number>(); // ip -> 上次成功上报时间戳

export default defineEventHandler(async (event) => {
  // mini 上 8081 直连暴露、无反代前置, XFF 可被客户端任意伪造, 只有 socket 地址可信
  if (getMethod(event) !== "POST") {
    setResponseHeader(event, "Allow", "POST");
    throw createError({ statusCode: 405, statusMessage: "Method Not Allowed" });
  }
  // 浏览器跨站校验: 有 Origin 头时必须同源, 防第三方页面用 text/plain 简单请求
  // (免预检)借访客浏览器代发; curl 等无 Origin 的非浏览器场景放行, 由 IP 限流兜底
  const origin = getRequestHeader(event, "origin");
  if (origin) {
    let originHost = "";
    try {
      originHost = new URL(origin).host;
    } catch {
      throw createError({ statusCode: 403, statusMessage: "Bad Origin" });
    }
    if (originHost !== getRequestHeader(event, "host")) {
      throw createError({ statusCode: 403, statusMessage: "Cross-Origin Not Allowed" });
    }
  }
  const ip = getRequestIP(event) || "unknown";
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

  console.log(`[report] 已转发 ip=${ip}: ${(userText || "(no description)").replace(/[\r\n\t]/g, " ").slice(0, 100)}`);
  // Map 防无限增长: 超过 1000 条时清掉已过窗口的旧条目
  if (lastReportAt.size > 1000) {
    for (const [k, t] of lastReportAt) {
      if (now - t >= RATE_LIMIT_MS) lastReportAt.delete(k);
    }
  }
  return { ok: true };
});
