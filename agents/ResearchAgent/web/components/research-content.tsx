"use client"

import { useState, useEffect, useRef } from "react"
import { ChevronDown, ChevronRight, Clock, CheckCircle, XCircle, Loader2, X, Lock, Unlock } from "lucide-react"
import { Button } from "@/components/ui/button"

// Type definitions
export type StepStatus = "IN_PROGRESS" | "COMPLETED" | "ERROR"

export type ContentType = "Markdown" | "Code" | "UrlList" | "Table" | "Image" | "Loader"

export interface ContentUnit {
  id: string
  content: any
  type: ContentType
}

export interface StepBlock {
  id: string
  title?: string
  status: StepStatus
  isCollapsible: boolean
  isCollapsed: boolean
  timestamp: string
  children: ContentUnit[]
}

// ContentUnit component
function ContentUnitRenderer({ unit }: { unit: ContentUnit }) {
  switch (unit.type) {
    case "Markdown":
      return (
        <div className="prose prose-sm prose-green max-w-none">
          <div className="text-green-300 font-mono text-sm whitespace-pre-wrap">{unit.content}</div>
        </div>
      )

    case "Code":
      return (
        <div className="bg-black/40 border border-green-500/20 rounded-lg p-4">
          <div className="text-green-400 text-xs font-mono mb-2">{unit.content.language}</div>
          <pre className="text-green-300 font-mono text-sm overflow-x-auto">
            <code>{unit.content.code}</code>
          </pre>
        </div>
      )

    case "UrlList":
      return (
        <div className="space-y-2">
          {unit.content.map((item: any, index: number) => (
            <div key={index} className="bg-black/20 border border-green-500/10 rounded-lg p-3">
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-green-400 hover:text-green-300 font-mono text-sm font-semibold"
              >
                {item.title}
              </a>
              {item.snippet && <p className="text-green-500 font-mono text-xs mt-1">{item.snippet}</p>}
            </div>
          ))}
        </div>
      )

    case "Table":
      return (
        <div className="overflow-x-auto">
          <table className="w-full border border-green-500/20 rounded-lg">
            <thead>
              <tr className="bg-green-500/10">
                {unit.content.headers.map((header: string, index: number) => (
                  <th
                    key={index}
                    className="border border-green-500/20 px-3 py-2 text-green-400 font-mono text-sm text-left"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {unit.content.rows.map((row: string[], rowIndex: number) => (
                <tr key={rowIndex}>
                  {row.map((cell: string, cellIndex: number) => (
                    <td
                      key={cellIndex}
                      className="border border-green-500/20 px-3 py-2 text-green-300 font-mono text-sm"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )

    case "Image":
      return (
        <div className="text-center">
          <img
            src={unit.content.url || "/placeholder.svg"}
            alt={unit.content.altText}
            className="max-w-full h-auto rounded-lg border border-green-500/20"
          />
          {unit.content.altText && <p className="text-green-500 font-mono text-xs mt-2">{unit.content.altText}</p>}
        </div>
      )

    case "Loader":
      return (
        <div className="flex items-center space-x-2 text-green-400">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="font-mono text-sm">{unit.content}</span>
        </div>
      )

    default:
      return <div className="text-green-500 font-mono text-sm">Unknown content type: {unit.type}</div>
  }
}

// StepBlock component
function StepBlockRenderer({ step }: { step: StepBlock }) {
  const [isCollapsed, setIsCollapsed] = useState(step.isCollapsed)

  const getStatusIcon = () => {
    switch (step.status) {
      case "IN_PROGRESS":
        return <Loader2 className="h-4 w-4 animate-spin text-yellow-400" />
      case "COMPLETED":
        return <CheckCircle className="h-4 w-4 text-green-400" />
      case "ERROR":
        return <XCircle className="h-4 w-4 text-red-400" />
    }
  }

  const getStatusColor = () => {
    switch (step.status) {
      case "IN_PROGRESS":
        return "border-yellow-500/30 bg-yellow-500/5"
      case "COMPLETED":
        return "border-green-500/30 bg-green-500/5"
      case "ERROR":
        return "border-red-500/30 bg-red-500/5"
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor()}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          {step.isCollapsible && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="h-6 w-6 p-0 text-green-400 hover:text-green-300"
            >
              {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          )}
          {getStatusIcon()}
          {step.title && <h3 className="text-green-400 font-mono font-semibold">{step.title}</h3>}
        </div>
        <div className="flex items-center space-x-1 text-green-500 text-xs font-mono">
          <Clock className="h-3 w-3" />
          <span>{formatTimestamp(step.timestamp)}</span>
        </div>
      </div>

      {/* Content */}
      {!isCollapsed && (
        <div className="grid grid-cols-1 gap-4">
          {step.children.map((unit) => (
            <div key={unit.id} className="bg-black/10 rounded-lg p-3">
              <ContentUnitRenderer unit={unit} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Main ResearchContent component
export function ResearchContent({
  triggerAnimation,
  onClose,
}: {
  triggerAnimation?: number
  onClose?: () => void
}) {
  const [allSteps, setAllSteps] = useState<StepBlock[]>([]) // Store all accumulated steps
  const [isAnimating, setIsAnimating] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true) // Auto-scroll toggle
  const [lastProcessedTrigger, setLastProcessedTrigger] = useState(0) // Track last processed trigger to avoid re-animation
  const [animatedStepIds, setAnimatedStepIds] = useState<Set<string>>(new Set())
  const [generatedStepsCache, setGeneratedStepsCache] = useState<Map<number, StepBlock[]>>(new Map())
  const [isInitialized, setIsInitialized] = useState(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null) // Scroll container reference

  // Test data - different sets for each message
  const getTestSteps = (messageIndex: number): StepBlock[] => {
    if (generatedStepsCache.has(messageIndex)) {
      return generatedStepsCache.get(messageIndex)!
    }

    const baseTime = Date.now() - messageIndex * 300000 // Different timestamps for each message set

    let steps: StepBlock[] = []

    if (messageIndex % 3 === 0) {
      steps = [
        {
          id: `msg${messageIndex}-step-1`,
          title: `分析查询 #${messageIndex + 1}`,
          status: "COMPLETED",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-1`,
              type: "Markdown",
              content: `开始处理用户查询 #${messageIndex + 1}：**${["人工智能发展", "机器学习算法", "深度学习应用"][messageIndex % 3]}**\n\n正在准备相关的研究资源...`,
            },
          ],
        },
        {
          id: `msg${messageIndex}-step-2`,
          title: "数据收集",
          status: "IN_PROGRESS",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime - 60000).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-2`,
              type: "Loader",
              content: "正在收集相关数据...",
            },
          ],
        },
      ]
    } else if (messageIndex % 3 === 1) {
      steps = [
        {
          id: `msg${messageIndex}-step-1`,
          title: `代码执行 #${messageIndex + 1}`,
          status: "COMPLETED",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-1`,
              type: "Code",
              content: {
                language: "python",
                code: `# 查询 #${messageIndex + 1}\nimport pandas as pd\n\ndata = pd.read_csv('research_data.csv')\nresult = data.analyze()\nprint(f"分析完成: {result}")`,
              },
            },
          ],
        },
        {
          id: `msg${messageIndex}-step-2`,
          title: "结果分析",
          status: "COMPLETED",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime - 60000).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-2`,
              type: "Table",
              content: {
                headers: ["指标", "数值", "状态"],
                rows: [
                  ["准确率", "95.2%", "优秀"],
                  ["处理速度", "1.2s", "良好"],
                  ["资源使用", "78%", "正常"],
                ],
              },
            },
          ],
        },
      ]
    } else {
      steps = [
        {
          id: `msg${messageIndex}-step-1`,
          title: `资源搜索 #${messageIndex + 1}`,
          status: "COMPLETED",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-1`,
              type: "UrlList",
              content: [
                {
                  url: `https://example.com/research-${messageIndex}`,
                  title: `研究资源 #${messageIndex + 1}`,
                  snippet: `相关研究文档和数据集...`,
                },
                {
                  url: `https://papers.com/paper-${messageIndex}`,
                  title: `学术论文 #${messageIndex + 1}`,
                  snippet: `最新的研究成果和发现...`,
                },
              ],
            },
          ],
        },
        {
          id: `msg${messageIndex}-step-2`,
          title: "可视化结果",
          status: "COMPLETED",
          isCollapsible: true,
          isCollapsed: false,
          timestamp: new Date(baseTime - 60000).toISOString(),
          children: [
            {
              id: `msg${messageIndex}-content-2`,
              type: "Image",
              content: {
                url: "/ai-------.jpg",
                altText: `分析结果图表 #${messageIndex + 1}`,
              },
            },
          ],
        },
      ]
    }

    setGeneratedStepsCache((prev) => new Map(prev).set(messageIndex, steps))
    return steps
  }

  const scrollToBottom = () => {
    if (autoScroll && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: "smooth",
      })
    }
  }

  useEffect(() => {
    console.log(
      "[v0] triggerAnimation useEffect triggered - triggerAnimation:",
      triggerAnimation,
      "lastProcessedTrigger:",
      lastProcessedTrigger,
    )

    if (!isInitialized) {
      console.log("[v0] Skipping trigger - not initialized yet")
      return
    }

    if (triggerAnimation && triggerAnimation > 0 && triggerAnimation > lastProcessedTrigger) {
      console.log("[v0] Processing new trigger:", triggerAnimation)
      const newSteps = getTestSteps(triggerAnimation - 1)
      console.log(
        "[v0] Generated steps for trigger:",
        newSteps.map((s) => s.id),
      )

      const existingStepIds = new Set(allSteps.map((step) => step.id))
      console.log("[v0] Existing step IDs:", Array.from(existingStepIds))

      const actuallyNewSteps = newSteps.filter((step) => !existingStepIds.has(step.id))
      console.log(
        "[v0] Actually new steps:",
        actuallyNewSteps.map((s) => s.id),
      )

      if (actuallyNewSteps.length > 0) {
        const hasExistingSteps = allSteps.length > 0
        console.log("[v0] Has existing steps:", hasExistingSteps)

        if (hasExistingSteps) {
          console.log("[v0] Starting sequential animation for new steps")
          setIsAnimating(true)

          const showStepsSequentially = async () => {
            for (let i = 0; i < actuallyNewSteps.length; i++) {
              console.log("[v0] Adding step with delay:", actuallyNewSteps[i].id)
              await new Promise((resolve) => setTimeout(resolve, 800))

              setAllSteps((prev) => {
                console.log("[v0] Adding step to allSteps:", actuallyNewSteps[i].id)
                return [...prev, actuallyNewSteps[i]]
              })
              setAnimatedStepIds((prev) => {
                const newSet = new Set([...prev, actuallyNewSteps[i].id])
                console.log("[v0] Updated animatedStepIds:", Array.from(newSet))
                return newSet
              })

              setTimeout(scrollToBottom, 100)
            }
            console.log("[v0] Sequential animation completed")
            setIsAnimating(false)
          }

          showStepsSequentially()
        } else {
          console.log("[v0] No existing steps, adding all at once")
          setAllSteps((prev) => [...prev, ...actuallyNewSteps])
          setAnimatedStepIds((prev) => new Set([...prev, ...actuallyNewSteps.map((step) => step.id)]))
          setTimeout(scrollToBottom, 100)
        }
      } else {
        console.log("[v0] No new steps to add")
      }

      console.log("[v0] Setting lastProcessedTrigger to:", triggerAnimation)
      setLastProcessedTrigger(triggerAnimation)
    } else {
      console.log("[v0] Skipping trigger - conditions not met")
    }
  }, [triggerAnimation, autoScroll, lastProcessedTrigger, allSteps, isInitialized])

  useEffect(() => {
    console.log(
      "[v0] Initial setup useEffect - allSteps.length:",
      allSteps.length,
      "lastProcessedTrigger:",
      lastProcessedTrigger,
      "triggerAnimation:",
      triggerAnimation,
    )

    if (!isInitialized) {
      if (triggerAnimation && triggerAnimation > 0) {
        // Restore all steps up to current trigger without animation
        console.log("[v0] Restoring state for triggerAnimation:", triggerAnimation)
        const allHistorySteps: StepBlock[] = []
        const allHistoryStepIds: string[] = []

        // Add initial steps (msg0)
        const initialSteps = getTestSteps(0)
        allHistorySteps.push(...initialSteps)
        allHistoryStepIds.push(...initialSteps.map((s) => s.id))

        // Add all subsequent steps up to current trigger
        for (let i = 1; i < triggerAnimation; i++) {
          const steps = getTestSteps(i)
          allHistorySteps.push(...steps)
          allHistoryStepIds.push(...steps.map((s) => s.id))
        }

        console.log("[v0] Restored steps:", allHistoryStepIds)
        setAllSteps(allHistorySteps)
        setLastProcessedTrigger(triggerAnimation)
        setAnimatedStepIds(new Set(allHistoryStepIds))
      } else {
        // First time initialization
        console.log("[v0] First time initialization")
        const initialSteps = getTestSteps(0)
        console.log(
          "[v0] Initial steps:",
          initialSteps.map((s) => s.id),
        )
        setAllSteps(initialSteps)
        setLastProcessedTrigger(1)
        setAnimatedStepIds(new Set(initialSteps.map((step) => step.id)))
      }
      setIsInitialized(true)
    }
  }, [triggerAnimation, isInitialized])

  const handleClose = () => {
    console.log("[v0] Close button clicked - current state:")
    console.log("[v0] - allSteps.length:", allSteps.length)
    console.log("[v0] - lastProcessedTrigger:", lastProcessedTrigger)
    console.log("[v0] - animatedStepIds:", Array.from(animatedStepIds))
    console.log("[v0] - generatedStepsCache:", Array.from(generatedStepsCache.keys()))

    if (onClose) {
      onClose()
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-green-500/20 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-green-400 font-mono">Content</h2>
        <div className="flex items-center space-x-2">
          <Button
            onClick={() => setAutoScroll(!autoScroll)}
            size="sm"
            variant="ghost"
            className={`h-8 w-8 p-0 ${autoScroll ? "text-green-400 hover:text-green-300" : "text-green-600 hover:text-green-500"} hover:bg-green-500/10`}
            title={autoScroll ? "取消自动滚动" : "开启自动滚动"}
          >
            {autoScroll ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
          </Button>
          {onClose && (
            <Button
              onClick={handleClose}
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0 text-green-400 hover:text-green-300 hover:bg-green-500/10"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Content with scroll */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-green-500/30 hover:scrollbar-thumb-green-500/50"
      >
        {allSteps.map((step, index) => (
          <div
            key={step.id}
            className={animatedStepIds.has(step.id) ? "" : "animate-in slide-in-from-top-4 fade-in-0 duration-500"}
            style={animatedStepIds.has(step.id) ? {} : { animationDelay: `${index * 100}ms` }}
          >
            <StepBlockRenderer step={step} />
          </div>
        ))}

        {isAnimating && (
          <div className="flex items-center justify-center p-8">
            <div className="flex items-center space-x-2 text-green-400">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="font-mono text-sm">Loading next step...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
