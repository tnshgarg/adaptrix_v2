"use client";

import { motion } from "framer-motion";
import {
  Terminal,
  Copy,
  Download,
  Zap,
  Brain,
  Database,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useState } from "react";

export function CLISection() {
  const [copied, setCopied] = useState(false);

  const installCommand =
    "curl -sSL https://raw.githubusercontent.com/adaptrix/adaptrix/main/install_adaptrix_cli.sh | bash";

  const copyToClipboard = () => {
    navigator.clipboard.writeText(installCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const cliFeatures = [
    {
      icon: Brain,
      title: "Model Management",
      description: "Download and run open-source models (<3B parameters)",
      command: "adaptrix models download qwen/qwen3-1.7b",
    },
    {
      icon: Zap,
      title: "Adapter Composition",
      description: "Install and combine multiple LoRA adapters",
      command: "adaptrix adapters install code_generator",
    },
    {
      icon: Database,
      title: "RAG Integration",
      description: "Add documents and create vector stores",
      command: "adaptrix rag add --collection docs ./documents",
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <section
      id="cli"
      className="py-24 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 border border-blue-200 dark:border-blue-800 mb-6">
            <Terminal className="w-4 h-4 text-blue-600 mr-2" />
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Powerful CLI Tool
            </span>
          </div>

          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="gradient-text">Command Your AI</span>
          </h2>

          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Install the Adaptrix CLI and get instant access to model management,
            adapter composition, and RAG integration from your terminal.
          </p>
        </motion.div>

        {/* Installation Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-16"
        >
          <Card className="max-w-4xl mx-auto bg-gray-900 dark:bg-gray-800 border-gray-700">
            <CardContent className="p-8">
              <div className="flex items-center justify-center mb-6">
                <Terminal className="w-8 h-8 text-green-400 mr-3" />
                <h3 className="text-2xl font-bold text-white">
                  One-Line Installation
                </h3>
              </div>

              <div className="bg-gray-800 dark:bg-gray-700 rounded-lg p-6 mb-6">
                <div className="flex items-center justify-between">
                  <code className="text-green-400 font-mono text-sm md:text-base flex-1 overflow-x-auto">
                    {installCommand}
                  </code>
                  <Button
                    onClick={copyToClipboard}
                    variant="ghost"
                    size="sm"
                    className="ml-4 text-gray-300 hover:text-white shrink-0"
                  >
                    {copied ? (
                      <>
                        <Download className="w-4 h-4 mr-2" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy
                      </>
                    )}
                  </Button>
                </div>
              </div>

              <div className="text-center">
                <p className="text-gray-400 text-sm mb-4">
                  Compatible with macOS, Linux, and Windows (WSL)
                </p>
                <div className="flex flex-wrap justify-center gap-4 text-xs">
                  <span className="px-3 py-1 bg-gray-700 rounded-full text-gray-300">
                    Python 3.8+
                  </span>
                  <span className="px-3 py-1 bg-gray-700 rounded-full text-gray-300">
                    Git Required
                  </span>
                  <span className="px-3 py-1 bg-gray-700 rounded-full text-gray-300">
                    Auto-Updates
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* CLI Features */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16"
        >
          {cliFeatures.map((feature, index) => (
            <motion.div key={index} variants={itemVariants}>
              <Card className="h-full group hover:shadow-xl transition-all duration-300 border-0 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    <div className="p-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 group-hover:scale-110 transition-transform duration-300">
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold ml-4 group-hover:text-primary transition-colors">
                      {feature.title}
                    </h3>
                  </div>

                  <p className="text-muted-foreground mb-4">
                    {feature.description}
                  </p>

                  <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
                    <code className="text-sm font-mono text-gray-700 dark:text-gray-300">
                      {feature.command}
                    </code>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Quick Start */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-center"
        >
          <Card className="max-w-4xl mx-auto">
            <CardContent className="p-8">
              <h3 className="text-2xl font-bold mb-6">Quick Start Guide</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-left">
                <div>
                  <h4 className="font-semibold mb-3 flex items-center">
                    <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      1
                    </span>
                    Install CLI
                  </h4>
                  <div className="bg-gray-100 dark:bg-gray-800 rounded p-3 mb-4">
                    <code className="text-sm">curl -sSL ... | bash</code>
                  </div>

                  <h4 className="font-semibold mb-3 flex items-center">
                    <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      2
                    </span>
                    Download Model
                  </h4>
                  <div className="bg-gray-100 dark:bg-gray-800 rounded p-3">
                    <code className="text-sm">
                      adaptrix models download qwen/qwen3-1.7b
                    </code>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3 flex items-center">
                    <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      3
                    </span>
                    Install Adapters
                  </h4>
                  <div className="bg-gray-100 dark:bg-gray-800 rounded p-3 mb-4">
                    <code className="text-sm">
                      adaptrix adapters install code_generator
                    </code>
                  </div>

                  <h4 className="font-semibold mb-3 flex items-center">
                    <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      4
                    </span>
                    Start Chatting
                  </h4>
                  <div className="bg-gray-100 dark:bg-gray-800 rounded p-3">
                    <code className="text-sm">
                      adaptrix chat --model qwen/qwen3-1.7b
                    </code>
                  </div>
                </div>
              </div>

              <div className="mt-8">
                <Button variant="outline" className="group">
                  View Full Documentation
                  <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
