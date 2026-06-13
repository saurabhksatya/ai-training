"use client";

import { Geist } from "next/font/google";
import { MdTune } from "react-icons/md";
import Link from "next/link";
import { usePathname } from "next/navigation";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="bg-surface-container border-b border-white/30 flex justify-between items-center w-full px-lg h-16 sticky top-0 z-50">
      <div className="flex items-center gap-sm">
        <span className="px-2">
          <MdTune size={24} />
        </span>
        <Link href="/" className="hover:opacity-85 transition-opacity">
          <h1
            className={`${geistSans.className} font-bold text-on-surface text-2xl`}
          >
            AI Training
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
      </nav>
    </header>
  );
}

