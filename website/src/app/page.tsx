import ClassificationReport from "@/components/Classification-Report";
import ConfusionMatrix from "@/components/Confusion-Matrix";
import Metrics from "@/components/Metric";
import Navbar from "@/components/Navbar";
import Selection from "@/components/selection";

export default function Home() {
  return (
    <>
      <Navbar />
      <main className="max-w-[1440px] flex flex-col gap-xl p-xl w-screen mx-auto">
        <Selection />
        <Metrics />
        <ClassificationReport />
        <ConfusionMatrix />
      </main>
    </>
  );
}
