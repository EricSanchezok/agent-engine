"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send } from "lucide-react"
import SplitText from "./split-text"

interface InitialInterfaceProps {
  onSendMessage: (message: string) => void
}

export function InitialInterface({ onSendMessage }: InitialInterfaceProps) {
  const [message, setMessage] = useState("")
  const [hasAnimated, setHasAnimated] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim()) {
      onSendMessage(message.trim())
      setMessage("")
    }
  }

  const handleAnimationComplete = () => {
    setHasAnimated(true)
    console.log("Title animation completed!")
  }

  return (
    <div className="min-h-screen flex flex-col animate-in fade-in-0 duration-500">
      <div className="absolute inset-0 bg-black/50 z-0" />

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-8">
        <div className="w-full max-w-2xl text-center space-y-8 animate-in slide-in-from-bottom-8 duration-700 delay-200">
          <div className="space-y-6">
            {!hasAnimated ? (
              <SplitText
                text="Researcher"
                className="text-6xl font-bold text-green-400 font-mono"
                tag="h1"
                delay={100}
                duration={0.6}
                ease="power3.out"
                splitType="chars"
                from={{ opacity: 0, y: 40 }}
                to={{ opacity: 1, y: 0 }}
                threshold={0.1}
                rootMargin="-100px"
                textAlign="center"
                onLetterAnimationComplete={handleAnimationComplete}
              />
            ) : (
              <h1 className="font-bold text-green-400 font-mono text-6xl">Researcher</h1>
            )}

            <p className="text-xl text-green-300/80 font-mono animate-in fade-in-0 slide-in-from-bottom-4 duration-500 delay-500">
              Your efficient research partner
            </p>
          </div>

          <div className="bg-black/40 backdrop-blur-md border border-green-500/20 rounded-2xl p-6 transition-all duration-500 hover:border-green-500/30 animate-in fade-in-0 slide-in-from-bottom-6 duration-700 delay-700 leading-7 w-auto">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Ask a question..."
                  className="w-full h-14 px-4 pr-12 text-lg bg-black/50 border border-green-500/30 rounded-xl focus:border-green-400 transition-colors font-mono placeholder:text-green-600 text-green-300 backdrop-blur-sm"
                />
                <Button
                  type="submit"
                  size="sm"
                  className="absolute right-2 top-1/2 -translate-y-1/2 h-10 w-10 p-0 bg-green-500 hover:bg-green-400 text-black"
                  disabled={!message.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </form>
          </div>
        </div>
      </main>
    </div>
  )
}
