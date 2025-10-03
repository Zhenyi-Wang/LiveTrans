import { computed, ref } from 'vue'

export function useParagraphLogic(confirmedSegments, maxParagraphLength) {
  // 第一步：将segments分成段落（统一分段逻辑）
  const getSegmentParagraphs = computed(() => {
    const paragraphs = []
    let currentSegments = []
    let currentLength = 0
    const maxLength = maxParagraphLength.value

    // 处理已确认的segments
    confirmedSegments.value.forEach(segment => {
      if (segment.text) {
        const segmentLength = (segment.opti_text || segment.text).length

        currentSegments.push(segment)
        currentLength += segmentLength

        if (currentLength >= maxLength) {
          // 寻找合适的分段点
          let splitIndex = currentSegments.length
          const sentenceEnds = ['。', '？', '！', '；', '.', '?', '!', ';']

          // 在当前段落中寻找分段点
          let accumulatedLength = 0

          for (let i = currentSegments.length - 1; i >= Math.max(0, currentSegments.length - 10); i--) {
            accumulatedLength += (currentSegments[i].opti_text || currentSegments[i].text).length
            if (currentLength - accumulatedLength <= maxLength - 50) {
              break
            }

            const segmentText = currentSegments[i].opti_text || currentSegments[i].text
            for (let j = segmentText.length - 1; j >= Math.max(0, segmentText.length - 50); j--) {
              if (sentenceEnds.includes(segmentText[j])) {
                splitIndex = i
                break
              }
            }
            if (splitIndex < currentSegments.length) break
          }

          // 如果没找到合适的分段点，就在当前segment处分段
          if (splitIndex === currentSegments.length) {
            splitIndex = currentSegments.length - 1
          }

          // 保存段落
          const paragraphSegments = currentSegments.slice(0, splitIndex)
          paragraphs.push(paragraphSegments)

          // 重置 - 计算剩余segments的长度
          const segmentLength = paragraphSegments.reduce((sum, seg) =>
            sum + (seg.opti_text || seg.text).length, 0)
          currentSegments = currentSegments.slice(splitIndex)
          currentLength -= segmentLength
        }
      }
    })

    // 处理剩余的segments
    if (currentSegments.length > 0) {
      paragraphs.push(currentSegments)
    }

    return paragraphs
  })

  // 第二步：根据分段生成中文段落
  const getChineseParagraphs = computed(() => {
    return getSegmentParagraphs.value.map(segmentGroup => {
      return segmentGroup.map(seg => ({
        text: seg.opti_text || seg.text,
        isOptimized: seg.opti_text && seg.opti_text !== seg.text
      }))
    })
  })

  // 第三步：根据分段生成英文段落
  const getEnglishParagraphs = computed(() => {
    return getSegmentParagraphs.value.map(segmentGroup => {
      const segments = segmentGroup.map(seg => ({
        text: seg.en_text || '',
        isTranslating: !seg.en_text, // 标记正在翻译的segment
        id: seg.id || `seg-${Math.random().toString(36).substr(2, 9)}`, // 确保有唯一id用于动画
        hasContent: !!seg.en_text // 是否有翻译内容
      }))

      return segments
    })
  })

  return {
    getSegmentParagraphs,
    getChineseParagraphs,
    getEnglishParagraphs
  }
}