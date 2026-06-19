/**
 * Naive UI theme overrides for Neumorphism.
 *
 * Strategy:
 * - common.bodyColor / common.cardColor = neumorphism base color
 * - Button/Card/etc. match surface color and use dual box-shadows
 * - Pressed/active states use inset shadows
 * - Accent (purple) reserved for focus rings and active indicators
 */
import type { GlobalThemeOverrides } from "naive-ui";

const SHADOW_RAISED = "8px 8px 16px #b8bcc8, -8px -8px 16px #ffffff";
const SHADOW_RAISED_SM =
  "4px 4px 8px #b8bcc8, -4px -4px 8px #ffffff";
const SHADOW_PRESSED =
  "inset 5px 5px 10px #b8bcc8, inset -5px -5px 10px #ffffff";
const SHADOW_FOCUS =
  "0 0 0 3px rgba(108, 99, 255, 0.3), 4px 4px 8px #b8bcc8, -4px -4px 8px #ffffff";

export const neumorphismOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: "#6c63ff",
    primaryColorHover: "#5a52e0",
    primaryColorPressed: "#4a43c0",
    primaryColorSuppl: "#6c63ff",
    bodyColor: "#e0e5ec",
    cardColor: "#e0e5ec",
    modalColor: "#e0e5ec",
    popoverColor: "#e0e5ec",
    tableColor: "#ffffff",
    inputColor: "#e0e5ec",
    actionColor: "#e0e5ec",
    tagColor: "#d8dde4",
    borderRadius: "14px",
    borderRadiusSmall: "10px",
    textColorBase: "#3d4852",
    textColor1: "#3d4852",
    textColor2: "#3d4852",
    textColor3: "#6b7280",
    placeholderColor: "#94a3b8",
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
    fontWeightStrong: "600",
  },
  Button: {
    fontWeight: "600",
    color: "#e0e5ec",
    colorHover: "#d8dde4",
    colorPressed: "#d8dde4",
    colorFocus: "#e0e5ec",
    colorDisabled: "#d8dde4",
    textColor: "#3d4852",
    textColorHover: "#3d4852",
    textColorPressed: "#3d4852",
    textColorFocus: "#3d4852",
    border: "none",
    borderHover: "none",
    borderPressed: "none",
    borderFocus: "none",
    boxShadow: SHADOW_RAISED,
    boxShadowHover: SHADOW_RAISED,
    boxShadowPressed: SHADOW_PRESSED,
    boxShadowFocus: SHADOW_FOCUS,
  },
  Card: {
    color: "#e0e5ec",
    borderColor: "transparent",
    boxShadow: SHADOW_RAISED,
    paddingMedium: "20px 24px",
  },
  Input: {
    color: "#e0e5ec",
    colorFocus: "#e0e5ec",
    border: "none",
    borderHover: "none",
    borderFocus: "none",
    boxShadow: SHADOW_PRESSED,
    boxShadowHover: SHADOW_PRESSED,
    boxShadowFocus: SHADOW_FOCUS,
    borderRadius: "10px",
  },
  Select: {
    peers: {
      InternalSelection: {
        border: "none",
        borderHover: "none",
        borderFocus: "none",
        borderRadius: "10px",
      },
    },
  },
  DataTable: {
    thColor: "#d8dde4",
    tdColor: "#ffffff",
    tdColorHover: "#f0f3f8",
    borderColor: "rgba(184, 188, 200, 0.3)",
    borderRadius: "14px",
    thFontWeight: "600",
  },
  Tag: {
    color: "#d8dde4",
    textColor: "#3d4852",
    border: "none",
    borderRadius: "10px",
  },
  Slider: {
    fillColor: "#6c63ff",
    fillColorHover: "#5a52e0",
    handleColor: "#6c63ff",
    railColor: "#ccd1d9",
  },
  Menu: {
    color: "#e0e5ec",
    itemColorHover: "#d8dde4",
    itemColorActive: "#d8dde4",
    itemColorActiveHover: "#ccd1d9",
    itemTextColor: "#3d4852",
    itemTextColorHover: "#3d4852",
    itemTextColorActive: "#6c63ff",
    itemTextColorActiveHover: "#6c63ff",
    borderRadius: "10px",
  },
  Tabs: {
    tabTextColorLine: "#6b7280",
    tabTextColorActiveLine: "#6c63ff",
    barColor: "#6c63ff",
  },
  Spin: {
    color: "#6c63ff",
  },
};
