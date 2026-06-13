"use client";

import { Geist } from "next/font/google";
import { MdTune } from "react-icons/md";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="bg-surface-container border-b border-white/30 flex justify-between items-center w-full px-lg h-16 sticky top-0 z-50">
      <div className="flex items-center gap-sm">
        <Image src="/logo.png" alt="logo" width={52} height={52} />
        <Link href="/" className="hover:opacity-85 transition-opacity -ml-2">
          <h1
            className={`${geistSans.className} font-bold text-on-surface text-2xl ml-0`}
          >
            NeuroForge
          </h1>
        </Link>
      </div>
      <nav className="flex items-center gap-md">
        <Link
          href="/"
          className={`text-sm font-medium transition-colors px-3 py-1.5 rounded-lg border ${
            pathname === "/"
              ? "text-primary bg-primary/10 border-primary/20"
              : "text-on-surface-variant hover:text-on-surface border-transparent"
          }`}
        >
          Classical Models
        </Link>
        <Link
          href="/neural-train"
          className={`text-sm font-medium transition-colors px-3 py-1.5 rounded-lg border ${
            pathname === "/neural-train"
              ? "text-primary bg-primary/10 border-primary/20"
              : "text-on-surface-variant hover:text-on-surface border-transparent"
          }`}
        >
          Neural Network Trainer
        </Link>
        <Link
          href="/rl-train"
          className={`text-sm font-medium transition-colors px-3 py-1.5 rounded-lg border ${
            pathname === "/rl-train"
              ? "text-primary bg-primary/10 border-primary/20"
              : "text-on-surface-variant hover:text-on-surface border-transparent"
          }`}
        >
          Reinforcement Learning Lab
        </Link>
      </nav>
    </header>
  );
}
