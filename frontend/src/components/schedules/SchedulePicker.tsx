"use client";

import { useState } from "react";

const PRESETS = [
  { label: "Every hour", expression: "0 * * * ? *" },
  { label: "Daily at 9am", expression: "0 9 * * ? *" },
  { label: "Every Monday", expression: "0 9 ? * MON *" },
  { label: "1st of month", expression: "0 9 1 * ? *" },
];

const TIMEZONES = [
  "UTC",
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Toronto",
  "Europe/London",
  "Europe/Paris",
  "Asia/Tokyo",
  "Asia/Shanghai",
  "Australia/Sydney",
];

interface SchedulePickerProps {
  expression: string;
  timezone: string;
  onExpressionChange: (expression: string) => void;
  onTimezoneChange: (timezone: string) => void;
}

export function SchedulePicker({
  expression,
  timezone,
  onExpressionChange,
  onTimezoneChange,
}: SchedulePickerProps) {
  const [isCustom, setIsCustom] = useState(
    !PRESETS.some((p) => p.expression === expression)
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset.expression}
            type="button"
            onClick={() => {
              onExpressionChange(preset.expression);
              setIsCustom(false);
            }}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              !isCustom && expression === preset.expression
                ? "border-zinc-500 bg-zinc-700/50 text-zinc-200"
                : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
            }`}
          >
            {preset.label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setIsCustom(true)}
          className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
            isCustom
              ? "border-zinc-500 bg-zinc-700/50 text-zinc-200"
              : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
          }`}
        >
          Custom
        </button>
      </div>
      {isCustom && (
        <div>
          <label htmlFor="cron-expression" className="mb-1.5 block text-xs font-medium text-zinc-400">
            Cron expression (EventBridge format)
          </label>
          <input
            id="cron-expression"
            type="text"
            value={expression}
            onChange={(e) => onExpressionChange(e.target.value)}
            placeholder="0 9 * * ? *"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
          />
        </div>
      )}
      <div>
        <label htmlFor="schedule-timezone" className="mb-1.5 block text-xs font-medium text-zinc-400">
          Timezone
        </label>
        <select
          id="schedule-timezone"
          value={timezone}
          onChange={(e) => onTimezoneChange(e.target.value)}
          className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
        >
          {TIMEZONES.map((tz) => (
            <option key={tz} value={tz}>
              {tz}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
