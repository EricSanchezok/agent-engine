"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, PanelRightOpen } from "lucide-react"
import TextType from "@/components/text-type"
import { ResearchContent } from "@/components/research-content"

interface Message {
  id: string
  content: string
  sender: "user" | "agent"
}

interface WorkspaceInterfaceProps {
  messages: Message[]
  onSendMessage: (message: string) => void
}

export function WorkspaceInterface({ messages, onSendMessage }: WorkspaceInterfaceProps) {
  const [newMessage, setNewMessage] = useState("")
  const [showContentPanel, setShowContentPanel] = useState(true)
  const [animationTrigger, setAnimationTrigger] = useState(0)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (newMessage.trim()) {
      onSendMessage(newMessage.trim())
      setNewMessage("")
      setAnimationTrigger((prev) => prev + 1)
    }
  }

  return (
    <div className="min-h-screen p-8 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/80 z-0" />

      <div className="relative z-10 w-3/4 h-[80vh] bg-black/40 backdrop-blur-md border border-green-500/20 rounded-2xl p-6 flex gap-6 animate-in fade-in-0 zoom-in-95 duration-700 slide-in-from-bottom-4">
        {!showContentPanel && (
          <Button
            onClick={() => setShowContentPanel(true)}
            size="sm"
            variant="ghost"
            className="absolute top-3 right-3 h-8 w-8 p-0 text-green-400 hover:text-green-300 hover:bg-green-500/10 z-10"
          >
            <PanelRightOpen className="h-4 w-4" />
          </Button>
        )}

        {/* Left Chat Panel */}
        <div
          className={`${showContentPanel ? "w-1/3" : "w-3/4 mx-auto"} flex flex-col transition-all duration-500 ease-in-out`}
        >
          <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-green-500/30 hover:scrollbar-thumb-green-500/50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`${message.sender === "user" ? "flex justify-end" : "flex justify-start"}`}
              >
                <div
                  className={`p-3 rounded-lg ${
                    message.sender === "user"
                      ? "bg-green-500/20 text-green-300 border border-green-500/30 max-w-fit"
                      : "bg-black/30 text-green-400 max-w-[80%]"
                  }`}
                >
                  {message.sender === "agent" ? (
                    <TextType
                      text={message.content}
                      as="p"
                      typingSpeed={25}
                      showCursor={true}
                      cursorCharacter="|"
                      loop={false}
                      className="text-sm leading-relaxed font-mono"
                    />
                  ) : (
                    <p className="text-sm leading-relaxed font-mono">{message.content}</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="bg-black/30 border border-green-500/20 rounded-xl p-3">
            <form onSubmit={handleSubmit}>
              <div className="relative">
                <Input
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Ask a question..."
                  className="w-full h-12 px-4 pr-12 bg-transparent border-none focus:ring-0 font-mono placeholder:text-green-600 text-green-300"
                />
                <Button
                  type="submit"
                  size="sm"
                  className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 bg-green-500 hover:bg-green-400 text-black"
                  disabled={!newMessage.trim()}
                >
                  <Send className="h-3 w-3" />
                </Button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Content Panel */}
        {showContentPanel && (
          <div className="w-2/3 bg-black/20 border border-green-500/20 rounded-xl relative transition-all duration-300 ease-in-out">
            <ResearchContent triggerAnimation={animationTrigger} onClose={() => setShowContentPanel(false)} />
          </div>
        )}
      </div>
    </div>
  )
}
