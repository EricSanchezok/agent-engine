"use client"

import { useState, useMemo, memo } from "react"
import { InitialInterface } from "@/components/initial-interface"
import { WorkspaceInterface } from "@/components/workspace-interface"
import FaultyTerminal from "@/components/faulty-terminal"
import PillNav from "@/components/pill-nav"

const MemoizedFaultyTerminal = memo(FaultyTerminal)

export default function ResearchAgentPage() {
  const [isExpanded, setIsExpanded] = useState(false)
  const [messages, setMessages] = useState<Array<{ id: string; content: string; sender: "user" | "agent" }>>([])

  const handleSendMessage = (message: string) => {
    const newMessage = {
      id: Date.now().toString(),
      content: message,
      sender: "user" as const,
    }

    setMessages((prev) => [...prev, newMessage])
    setIsExpanded(true)

    // Simulate agent response
    setTimeout(() => {
      const agentResponse = {
        id: (Date.now() + 1).toString(),
        content: `I understand you're asking about: "${message}". Let me research this for you and provide a comprehensive analysis.`,
        sender: "agent" as const,
      }
      setMessages((prev) => [...prev, agentResponse])
    }, 1000)
  }

  const handleHomeClick = () => {
    setIsExpanded(false)
    setMessages([])
  }

  const handleAboutClick = () => {
    // For now, just show an alert - can be expanded later
    alert("About section - Coming soon!")
  }

  const handleContactClick = () => {
    // For now, just show an alert - can be expanded later
    alert("Contact section - Coming soon!")
  }

  const faultyTerminalProps = useMemo(
    () => ({
      scale: 1.8,
      gridMul: [2.5, 1.2] as [number, number],
      digitSize: 1.4,
      timeScale: 0.8,
      pause: false,
      scanlineIntensity: 0.8,
      glitchAmount: 1.2,
      flickerAmount: 0.6,
      noiseAmp: 1.5,
      chromaticAberration: 0.5,
      dither: 0.3,
      curvature: 0.15,
      tint: "#00ff41",
      mouseReact: true,
      mouseStrength: 0.7,
      pageLoadAnimation: false,
      brightness: 0.4,
    }),
    [], // Removed isExpanded dependency to prevent re-rendering
  )

  return (
    <div className="min-h-screen bg-background relative">
      <div className="fixed inset-0 z-0">
        <MemoizedFaultyTerminal key="background-terminal" {...faultyTerminalProps} />
      </div>

      <PillNav
        logo="/sunrise.png"
        logoAlt="ResearchAgent Logo"
        items={[
          { label: "HOME", href: "#", onClick: handleHomeClick },
          { label: "ABOUT", href: "#", onClick: handleAboutClick },
          { label: "CONTACT", href: "#", onClick: handleContactClick },
        ]}
        activeHref="#"
        baseColor="#00ff41"
        pillColor="#000000"
        hoveredPillTextColor="#000000"
        pillTextColor="#00ff41"
        className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50"
      />

      <div className="relative z-10">
        {!isExpanded ? (
          <InitialInterface onSendMessage={handleSendMessage} />
        ) : (
          <WorkspaceInterface messages={messages} onSendMessage={handleSendMessage} />
        )}
      </div>
    </div>
  )
}
