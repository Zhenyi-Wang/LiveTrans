// composables/useDemoFeed.ts
// /st 测试页专用:本地模拟直播数据流(current 分次演进 → confirmed 追加 → 翻译回填),
// 消息结构与真实链路一致(useWebSocket 的 current/confirmed/update 三类事件),无需真实推流。
// demo 只是叠加数据源:有真实直播时开 demo 会混入模拟数据,仅测试时使用。
import { ref, onBeforeUnmount, type Ref } from 'vue'

interface DemoSentence { zh: string; en: string }

const SENTENCES: DemoSentence[] = [
  { zh: '我们在天上的父，愿人都尊你的名为圣。', en: 'Our Father in heaven, hallowed be your name.' },
  { zh: '愿你的国降临，愿你的旨意行在地上，如同行在天上。', en: 'Your kingdom come, your will be done on earth as it is in heaven.' },
  { zh: '我们日用的饮食，今日赐给我们。', en: 'Give us today our daily bread.' },
  { zh: '免我们的债，如同我们免了人的债。', en: 'And forgive us our debts, as we also have forgiven our debtors.' },
  { zh: '不叫我们遇见试探，救我们脱离凶恶。', en: 'And lead us not into temptation, but deliver us from the evil one.' },
  { zh: '因为国度、权柄、荣耀，全是你的，直到永远。', en: 'For yours is the kingdom, the power, and the glory forever.' },
  { zh: '耶稣走遍各城各乡，在会堂里教训人，宣讲天国的福音，又医治各样的病症。', en: 'Jesus went through all the towns and villages, teaching in their synagogues, proclaiming the good news of the kingdom and healing every disease.' },
  { zh: '他看见许多的人，就怜悯他们，因为他们困苦流离，如同羊没有牧人一般。', en: 'When he saw the crowds, he had compassion on them, because they were harassed and helpless, like sheep without a shepherd.' },
  { zh: '于是对门徒说，要收的庄稼多，作工的人少。', en: 'Then he said to his disciples, the harvest is plentiful but the workers are few.' },
  { zh: '所以你们当求庄稼的主，打发工人出去收他的庄稼。', en: 'Ask the Lord of the harvest, therefore, to send out workers into his harvest field.' },
  { zh: '凡劳苦担重担的人，可以到我这里来，我就使你们得安息。', en: 'Come to me, all you who are weary and burdened, and I will give you rest.' },
  { zh: '我心里柔和谦卑，你们当负我的轭，学我的样式，这样你们心里就必得享安息。', en: 'Take my yoke upon you and learn from me, for I am gentle and humble in heart, and you will find rest for your souls.' }
]

const rand = (min: number, max: number) => min + Math.random() * (max - min)

export function useDemoFeed(
  currentSegment: Ref<{ text: string }>,
  lastCurrentEn: Ref<string>,
  confirmedSegments: { value: any[] }
) {
  const demoOn = ref(false)
  let timer: ReturnType<typeof setTimeout> | null = null
  let seq = 0
  let idx = 0

  const schedule = (fn: () => void, delay: number) => {
    timer = setTimeout(() => { if (demoOn.value) fn() }, delay)
  }

  // 一句的完整生命周期:current 分 3~6 次演进 → 追加进 confirmed(英文栏 translating) → 延迟回填翻译 → 下一句
  const playSentence = () => {
    const s = SENTENCES[idx % SENTENCES.length]
    idx++
    const id = `demo-${++seq}`
    const step = Math.ceil(s.zh.length / (3 + Math.floor(Math.random() * 4)))
    const chunks: string[] = []
    for (let pos = 0; pos < s.zh.length; pos += step) chunks.push(s.zh.slice(pos, pos + step))

    const typeChunk = (i: number) => {
      currentSegment.value = { text: currentSegment.value.text + chunks[i] }
      if (i === chunks.length - 1) {
        lastCurrentEn.value = s.en              // 模拟 current 英文预览
        schedule(confirm, rand(600, 1000))
      } else {
        schedule(() => typeChunk(i + 1), rand(300, 600))
      }
    }

    const confirm = () => {
      currentSegment.value = { text: '' }
      lastCurrentEn.value = ''
      confirmedSegments.value.push({ id, text: s.zh })   // 先无 en_text(英文栏显示 translating…)
      schedule(() => {
        // 翻译回填:走 update 原地替换,与真实矫正/翻译链路同构
        // (segment间空格由 ParagraphDisplayV2 渲染层显式输出,数据层保持干净)
        const i = confirmedSegments.value.findLastIndex(x => x.id === id)
        if (i >= 0) confirmedSegments.value[i] = { id, text: s.zh, en_text: s.en, opti_text: s.zh }
        schedule(playSentence, rand(500, 1500))
      }, rand(800, 2000))
    }

    typeChunk(0)
  }

  const toggleDemo = () => {
    demoOn.value = !demoOn.value
    if (demoOn.value) {
      playSentence()
    } else {
      if (timer) { clearTimeout(timer); timer = null }
      currentSegment.value = { text: '' }     // 关闭即清场,恢复干净状态
      lastCurrentEn.value = ''
      confirmedSegments.value = []
    }
    return demoOn.value
  }

  onBeforeUnmount(() => {
    if (timer) clearTimeout(timer)
  })

  return { demoOn, toggleDemo }
}
