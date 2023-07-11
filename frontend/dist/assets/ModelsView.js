import { Q as replaceable, E as h, R as createInjectionKey, d as defineComponent, S as inject, T as throwError, U as isBrowser, c as computed, V as resolveSlot, W as NBaseIcon, X as AddIcon, Y as NProgress, Z as NFadeInExpandTransition, F as ref, $ as useMemo, a0 as watchEffect, G as NButton, a1 as EyeIcon, a2 as NIconSwitchTransition, a3 as warn, a4 as c, a5 as cB, a6 as cM, a7 as fadeInHeightExpandTransition, a8 as cE, a9 as iconSwitchTransition, aa as useConfig, ab as useTheme, ac as useFormItem, ad as toRef, ae as useMergedState, af as useThemeClass, ag as provide, ah as Teleport, M as Fragment, ai as uploadLight, aj as createId, ak as nextTick, al as call, e as openBlock, f as createElementBlock, n as createBaseVNode, x as createBlock, w as withCtx, g as createVNode, h as unref, H as NIcon, am as NResult, u as useState, b as useMessage, an as reactive, ao as huggingfaceModelsFile, k as NInput, m as createTextVNode, i as NCard, s as serverUrl, ap as NDropdown, B as pushScopeId, C as popScopeId, _ as _export_sfc, r as NSelect, j as NSpace, aq as NModal, N as NGi, ar as NText, O as renderList, t as toDisplayString, z as NGrid, I as NTabPane, J as NTabs } from "./index.js";
import { N as NDataTable } from "./DataTable.js";
import { N as NSwitch } from "./Switch.js";
import { N as NImage, a as NImageGroup, T as TrashBin } from "./TrashBin.js";
import { C as CloudUpload } from "./CloudUpload.js";
const AttachIcon = replaceable("attach", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M3.25735931,8.70710678 L7.85355339,4.1109127 C8.82986412,3.13460197 10.4127766,3.13460197 11.3890873,4.1109127 C12.365398,5.08722343 12.365398,6.67013588 11.3890873,7.64644661 L6.08578644,12.9497475 C5.69526215,13.3402718 5.06209717,13.3402718 4.67157288,12.9497475 C4.28104858,12.5592232 4.28104858,11.9260582 4.67157288,11.5355339 L9.97487373,6.23223305 C10.1701359,6.0369709 10.1701359,5.72038841 9.97487373,5.52512627 C9.77961159,5.32986412 9.4630291,5.32986412 9.26776695,5.52512627 L3.96446609,10.8284271 C3.18341751,11.6094757 3.18341751,12.8758057 3.96446609,13.6568542 C4.74551468,14.4379028 6.01184464,14.4379028 6.79289322,13.6568542 L12.0961941,8.35355339 C13.4630291,6.98671837 13.4630291,4.77064094 12.0961941,3.40380592 C10.7293591,2.0369709 8.51328163,2.0369709 7.14644661,3.40380592 L2.55025253,8 C2.35499039,8.19526215 2.35499039,8.51184464 2.55025253,8.70710678 C2.74551468,8.90236893 3.06209717,8.90236893 3.25735931,8.70710678 Z" })
    )
  )
));
const TrashIcon = replaceable("trash", h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 512 512" },
  h("path", { d: "M432,144,403.33,419.74A32,32,0,0,1,371.55,448H140.46a32,32,0,0,1-31.78-28.26L80,144", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("rect", { x: "32", y: "64", width: "448", height: "80", rx: "16", ry: "16", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("line", { x1: "312", y1: "240", x2: "200", y2: "352", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" }),
  h("line", { x1: "312", y1: "352", x2: "200", y2: "240", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" })
));
const DownloadIcon = replaceable("download", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M3.5,13 L12.5,13 C12.7761424,13 13,13.2238576 13,13.5 C13,13.7454599 12.8231248,13.9496084 12.5898756,13.9919443 L12.5,14 L3.5,14 C3.22385763,14 3,13.7761424 3,13.5 C3,13.2545401 3.17687516,13.0503916 3.41012437,13.0080557 L3.5,13 L12.5,13 L3.5,13 Z M7.91012437,1.00805567 L8,1 C8.24545989,1 8.44960837,1.17687516 8.49194433,1.41012437 L8.5,1.5 L8.5,10.292 L11.1819805,7.6109127 C11.3555469,7.43734635 11.6249713,7.4180612 11.8198394,7.55305725 L11.8890873,7.6109127 C12.0626536,7.78447906 12.0819388,8.05390346 11.9469427,8.2487716 L11.8890873,8.31801948 L8.35355339,11.8535534 C8.17998704,12.0271197 7.91056264,12.0464049 7.7156945,11.9114088 L7.64644661,11.8535534 L4.1109127,8.31801948 C3.91565056,8.12275734 3.91565056,7.80617485 4.1109127,7.6109127 C4.28447906,7.43734635 4.55390346,7.4180612 4.7487716,7.55305725 L4.81801948,7.6109127 L7.5,10.292 L7.5,1.5 C7.5,1.25454011 7.67687516,1.05039163 7.91012437,1.00805567 L8,1 L7.91012437,1.00805567 Z" })
    )
  )
));
const CancelIcon = replaceable("cancel", h(
  "svg",
  { viewBox: "0 0 16 16", version: "1.1", xmlns: "http://www.w3.org/2000/svg" },
  h(
    "g",
    { stroke: "none", "stroke-width": "1", fill: "none", "fill-rule": "evenodd" },
    h(
      "g",
      { fill: "currentColor", "fill-rule": "nonzero" },
      h("path", { d: "M2.58859116,2.7156945 L2.64644661,2.64644661 C2.82001296,2.47288026 3.08943736,2.45359511 3.2843055,2.58859116 L3.35355339,2.64644661 L8,7.293 L12.6464466,2.64644661 C12.8417088,2.45118446 13.1582912,2.45118446 13.3535534,2.64644661 C13.5488155,2.84170876 13.5488155,3.15829124 13.3535534,3.35355339 L8.707,8 L13.3535534,12.6464466 C13.5271197,12.820013 13.5464049,13.0894374 13.4114088,13.2843055 L13.3535534,13.3535534 C13.179987,13.5271197 12.9105626,13.5464049 12.7156945,13.4114088 L12.6464466,13.3535534 L8,8.707 L3.35355339,13.3535534 C3.15829124,13.5488155 2.84170876,13.5488155 2.64644661,13.3535534 C2.45118446,13.1582912 2.45118446,12.8417088 2.64644661,12.6464466 L7.293,8 L2.64644661,3.35355339 C2.47288026,3.17998704 2.45359511,2.91056264 2.58859116,2.7156945 L2.64644661,2.64644661 L2.58859116,2.7156945 Z" })
    )
  )
));
const RetryIcon = replaceable("retry", h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 512 512" },
  h("path", { d: "M320,146s24.36-12-64-12A160,160,0,1,0,416,294", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-miterlimit: 10; stroke-width: 32px;" }),
  h("polyline", { points: "256 58 336 138 256 218", style: "fill: none; stroke: currentcolor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 32px;" })
));
const uploadInjectionKey = createInjectionKey("n-upload");
const uploadDraggerKey = "__UPLOAD_DRAGGER__";
const NUploadDragger = defineComponent({
  name: "UploadDragger",
  [uploadDraggerKey]: true,
  setup(_, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-dragger", "`n-upload-dragger` must be placed inside `n-upload`.");
    }
    return () => {
      const { mergedClsPrefixRef: { value: mergedClsPrefix }, mergedDisabledRef: { value: mergedDisabled }, maxReachedRef: { value: maxReached } } = NUpload2;
      return h("div", { class: [
        `${mergedClsPrefix}-upload-dragger`,
        (mergedDisabled || maxReached) && `${mergedClsPrefix}-upload-dragger--disabled`
      ] }, slots);
    };
  }
});
var __awaiter$2 = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
const isImageFileType = (type) => type.includes("image/");
const getExtname = (url = "") => {
  const temp = url.split("/");
  const filename = temp[temp.length - 1];
  const filenameWithoutSuffix = filename.split(/#|\?/)[0];
  return (/\.[^./\\]*$/.exec(filenameWithoutSuffix) || [""])[0];
};
const imageExtensionRegex = /(webp|svg|png|gif|jpg|jpeg|jfif|bmp|dpg|ico)$/i;
const isImageFile = (file) => {
  if (file.type) {
    return isImageFileType(file.type);
  }
  const fileNameExtension = getExtname(file.name || "");
  if (imageExtensionRegex.test(fileNameExtension)) {
    return true;
  }
  const url = file.thumbnailUrl || file.url || "";
  const urlExtension = getExtname(url);
  if (/^data:image\//.test(url) || imageExtensionRegex.test(urlExtension)) {
    return true;
  }
  return false;
};
function createImageDataUrl(file) {
  return __awaiter$2(this, void 0, void 0, function* () {
    return yield new Promise((resolve) => {
      if (!file.type || !isImageFileType(file.type)) {
        resolve("");
        return;
      }
      resolve(window.URL.createObjectURL(file));
    });
  });
}
const environmentSupportFile = isBrowser && window.FileReader && window.File;
function isFileSystemDirectoryEntry(item) {
  return item.isDirectory;
}
function isFileSystemFileEntry(item) {
  return item.isFile;
}
function getFilesFromEntries(entries, directory) {
  return __awaiter$2(this, void 0, void 0, function* () {
    const fileAndEntries = [];
    let _resolve;
    let requestCallbackCount = 0;
    function lock() {
      requestCallbackCount++;
    }
    function unlock() {
      requestCallbackCount--;
      if (!requestCallbackCount) {
        _resolve(fileAndEntries);
      }
    }
    function _getFilesFromEntries(entries2) {
      entries2.forEach((entry) => {
        if (!entry)
          return;
        lock();
        if (directory && isFileSystemDirectoryEntry(entry)) {
          const directoryReader = entry.createReader();
          lock();
          directoryReader.readEntries((entries3) => {
            _getFilesFromEntries(entries3);
            unlock();
          }, () => {
            unlock();
          });
        } else if (isFileSystemFileEntry(entry)) {
          lock();
          entry.file((file) => {
            fileAndEntries.push({ file, entry, source: "dnd" });
            unlock();
          }, () => {
            unlock();
          });
        }
        unlock();
      });
    }
    yield new Promise((resolve) => {
      _resolve = resolve;
      _getFilesFromEntries(entries);
    });
    return fileAndEntries;
  });
}
function createSettledFileInfo(fileInfo) {
  const { id, name, percentage, status, url, file, thumbnailUrl, type, fullPath, batchId } = fileInfo;
  return {
    id,
    name,
    percentage: percentage !== null && percentage !== void 0 ? percentage : null,
    status,
    url: url !== null && url !== void 0 ? url : null,
    file: file !== null && file !== void 0 ? file : null,
    thumbnailUrl: thumbnailUrl !== null && thumbnailUrl !== void 0 ? thumbnailUrl : null,
    type: type !== null && type !== void 0 ? type : null,
    fullPath: fullPath !== null && fullPath !== void 0 ? fullPath : null,
    batchId: batchId !== null && batchId !== void 0 ? batchId : null
  };
}
function matchType(name, mimeType, accept) {
  name = name.toLowerCase();
  mimeType = mimeType.toLocaleLowerCase();
  accept = accept.toLocaleLowerCase();
  const acceptAtoms = accept.split(",").map((acceptAtom) => acceptAtom.trim()).filter(Boolean);
  return acceptAtoms.some((acceptAtom) => {
    if (acceptAtom.startsWith(".")) {
      if (name.endsWith(acceptAtom))
        return true;
    } else if (acceptAtom.includes("/")) {
      const [type, subtype] = mimeType.split("/");
      const [acceptType, acceptSubtype] = acceptAtom.split("/");
      if (acceptType === "*" || type && acceptType && acceptType === type) {
        if (acceptSubtype === "*" || subtype && acceptSubtype && acceptSubtype === subtype) {
          return true;
        }
      }
    } else {
      return true;
    }
    return false;
  });
}
const download = (url, name) => {
  if (!url)
    return;
  const a = document.createElement("a");
  a.href = url;
  if (name !== void 0) {
    a.download = name;
  }
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};
const NUploadTrigger = defineComponent({
  name: "UploadTrigger",
  props: {
    abstract: Boolean
  },
  setup(props, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-trigger", "`n-upload-trigger` must be placed inside `n-upload`.");
    }
    const { mergedClsPrefixRef, mergedDisabledRef, maxReachedRef, listTypeRef, dragOverRef, openOpenFileDialog, draggerInsideRef, handleFileAddition, mergedDirectoryDndRef, triggerStyleRef } = NUpload2;
    const isImageCardTypeRef = computed(() => listTypeRef.value === "image-card");
    function handleTriggerClick() {
      if (mergedDisabledRef.value || maxReachedRef.value)
        return;
      openOpenFileDialog();
    }
    function handleTriggerDragOver(e) {
      e.preventDefault();
      dragOverRef.value = true;
    }
    function handleTriggerDragEnter(e) {
      e.preventDefault();
      dragOverRef.value = true;
    }
    function handleTriggerDragLeave(e) {
      e.preventDefault();
      dragOverRef.value = false;
    }
    function handleTriggerDrop(e) {
      var _a;
      e.preventDefault();
      if (!draggerInsideRef.value || mergedDisabledRef.value || maxReachedRef.value) {
        dragOverRef.value = false;
        return;
      }
      const dataTransferItems = (_a = e.dataTransfer) === null || _a === void 0 ? void 0 : _a.items;
      if (dataTransferItems === null || dataTransferItems === void 0 ? void 0 : dataTransferItems.length) {
        void getFilesFromEntries(Array.from(dataTransferItems).map((item) => item.webkitGetAsEntry()), mergedDirectoryDndRef.value).then((files) => {
          handleFileAddition(files);
        }).finally(() => {
          dragOverRef.value = false;
        });
      } else {
        dragOverRef.value = false;
      }
    }
    return () => {
      var _a;
      const { value: mergedClsPrefix } = mergedClsPrefixRef;
      return props.abstract ? (_a = slots.default) === null || _a === void 0 ? void 0 : _a.call(slots, {
        handleClick: handleTriggerClick,
        handleDrop: handleTriggerDrop,
        handleDragOver: handleTriggerDragOver,
        handleDragEnter: handleTriggerDragEnter,
        handleDragLeave: handleTriggerDragLeave
      }) : h("div", { class: [
        `${mergedClsPrefix}-upload-trigger`,
        (mergedDisabledRef.value || maxReachedRef.value) && `${mergedClsPrefix}-upload-trigger--disabled`,
        isImageCardTypeRef.value && `${mergedClsPrefix}-upload-trigger--image-card`
      ], style: triggerStyleRef.value, onClick: handleTriggerClick, onDrop: handleTriggerDrop, onDragover: handleTriggerDragOver, onDragenter: handleTriggerDragEnter, onDragleave: handleTriggerDragLeave }, isImageCardTypeRef.value ? h(NUploadDragger, null, {
        default: () => resolveSlot(slots.default, () => [
          h(NBaseIcon, { clsPrefix: mergedClsPrefix }, { default: () => h(AddIcon, null) })
        ])
      }) : slots);
    };
  }
});
const NUploadProgress = defineComponent({
  name: "UploadProgress",
  props: {
    show: Boolean,
    percentage: {
      type: Number,
      required: true
    },
    status: {
      type: String,
      required: true
    }
  },
  setup() {
    const NUpload2 = inject(uploadInjectionKey);
    return {
      mergedTheme: NUpload2.mergedThemeRef
    };
  },
  render() {
    return h(NFadeInExpandTransition, null, {
      default: () => this.show ? h(NProgress, { type: "line", showIndicator: false, percentage: this.percentage, status: this.status, height: 2, theme: this.mergedTheme.peers.Progress, themeOverrides: this.mergedTheme.peerOverrides.Progress }) : null
    });
  }
});
const imageIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 28 28" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M21.75 3A3.25 3.25 0 0 1 25 6.25v15.5A3.25 3.25 0 0 1 21.75 25H6.25A3.25 3.25 0 0 1 3 21.75V6.25A3.25 3.25 0 0 1 6.25 3h15.5zm.583 20.4l-7.807-7.68a.75.75 0 0 0-.968-.07l-.084.07l-7.808 7.68c.183.065.38.1.584.1h15.5c.204 0 .4-.035.583-.1l-7.807-7.68l7.807 7.68zM21.75 4.5H6.25A1.75 1.75 0 0 0 4.5 6.25v15.5c0 .208.036.408.103.593l7.82-7.692a2.25 2.25 0 0 1 3.026-.117l.129.117l7.82 7.692c.066-.185.102-.385.102-.593V6.25a1.75 1.75 0 0 0-1.75-1.75zm-3.25 3a2.5 2.5 0 1 1 0 5a2.5 2.5 0 0 1 0-5zm0 1.5a1 1 0 1 0 0 2a1 1 0 0 0 0-2z", fill: "currentColor" })
  )
);
const documentIcon = h(
  "svg",
  { xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 28 28" },
  h(
    "g",
    { fill: "none" },
    h("path", { d: "M6.4 2A2.4 2.4 0 0 0 4 4.4v19.2A2.4 2.4 0 0 0 6.4 26h15.2a2.4 2.4 0 0 0 2.4-2.4V11.578c0-.729-.29-1.428-.805-1.944l-6.931-6.931A2.4 2.4 0 0 0 14.567 2H6.4zm-.9 2.4a.9.9 0 0 1 .9-.9H14V10a2 2 0 0 0 2 2h6.5v11.6a.9.9 0 0 1-.9.9H6.4a.9.9 0 0 1-.9-.9V4.4zm16.44 6.1H16a.5.5 0 0 1-.5-.5V4.06l6.44 6.44z", fill: "currentColor" })
  )
);
var __awaiter$1 = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
const buttonThemeOverrides = {
  paddingMedium: "0 3px",
  heightMedium: "24px",
  iconSizeMedium: "18px"
};
const NUploadFile = defineComponent({
  name: "UploadFile",
  props: {
    clsPrefix: {
      type: String,
      required: true
    },
    file: {
      type: Object,
      required: true
    },
    listType: {
      type: String,
      required: true
    }
  },
  setup(props) {
    const NUpload2 = inject(uploadInjectionKey);
    const imageRef = ref(null);
    const thumbnailUrlRef = ref("");
    const progressStatusRef = computed(() => {
      const { file } = props;
      if (file.status === "finished")
        return "success";
      if (file.status === "error")
        return "error";
      return "info";
    });
    const buttonTypeRef = computed(() => {
      const { file } = props;
      if (file.status === "error")
        return "error";
      return void 0;
    });
    const showProgressRef = computed(() => {
      const { file } = props;
      return file.status === "uploading";
    });
    const showCancelButtonRef = computed(() => {
      if (!NUpload2.showCancelButtonRef.value)
        return false;
      const { file } = props;
      return ["uploading", "pending", "error"].includes(file.status);
    });
    const showRemoveButtonRef = computed(() => {
      if (!NUpload2.showRemoveButtonRef.value)
        return false;
      const { file } = props;
      return ["finished"].includes(file.status);
    });
    const showDownloadButtonRef = computed(() => {
      if (!NUpload2.showDownloadButtonRef.value)
        return false;
      const { file } = props;
      return ["finished"].includes(file.status);
    });
    const showRetryButtonRef = computed(() => {
      if (!NUpload2.showRetryButtonRef.value)
        return false;
      const { file } = props;
      return ["error"].includes(file.status);
    });
    const mergedThumbnailUrlRef = useMemo(() => {
      return thumbnailUrlRef.value || props.file.thumbnailUrl || props.file.url;
    });
    const showPreviewButtonRef = computed(() => {
      if (!NUpload2.showPreviewButtonRef.value)
        return false;
      const { file: { status }, listType } = props;
      return ["finished"].includes(status) && mergedThumbnailUrlRef.value && listType === "image-card";
    });
    function handleRetryClick() {
      NUpload2.submit(props.file.id);
    }
    function handleRemoveOrCancelClick(e) {
      e.preventDefault();
      const { file } = props;
      if (["finished", "pending", "error"].includes(file.status)) {
        handleRemove(file);
      } else if (["uploading"].includes(file.status)) {
        handleAbort(file);
      } else {
        warn("upload", "The button clicked type is unknown.");
      }
    }
    function handleDownloadClick(e) {
      e.preventDefault();
      handleDownload(props.file);
    }
    function handleRemove(file) {
      const { xhrMap, doChange, onRemoveRef: { value: onRemove }, mergedFileListRef: { value: mergedFileList } } = NUpload2;
      void Promise.resolve(onRemove ? onRemove({
        file: Object.assign({}, file),
        fileList: mergedFileList
      }) : true).then((result) => {
        if (result === false)
          return;
        const fileAfterChange = Object.assign({}, file, {
          status: "removed"
        });
        xhrMap.delete(file.id);
        doChange(fileAfterChange, void 0, {
          remove: true
        });
      });
    }
    function handleDownload(file) {
      const { onDownloadRef: { value: onDownload } } = NUpload2;
      void Promise.resolve(onDownload ? onDownload(Object.assign({}, file)) : true).then((res) => {
        if (res !== false) {
          download(file.url, file.name);
        }
      });
    }
    function handleAbort(file) {
      const { xhrMap } = NUpload2;
      const xhr = xhrMap.get(file.id);
      xhr === null || xhr === void 0 ? void 0 : xhr.abort();
      handleRemove(Object.assign({}, file));
    }
    function handlePreviewClick() {
      const { onPreviewRef: { value: onPreview } } = NUpload2;
      if (onPreview) {
        onPreview(props.file);
      } else if (props.listType === "image-card") {
        const { value } = imageRef;
        if (!value)
          return;
        value.click();
      }
    }
    const deriveFileThumbnailUrl = () => __awaiter$1(this, void 0, void 0, function* () {
      const { listType } = props;
      if (listType !== "image" && listType !== "image-card") {
        return;
      }
      if (NUpload2.shouldUseThumbnailUrlRef.value(props.file)) {
        thumbnailUrlRef.value = yield NUpload2.getFileThumbnailUrlResolver(props.file);
      }
    });
    watchEffect(() => {
      void deriveFileThumbnailUrl();
    });
    return {
      mergedTheme: NUpload2.mergedThemeRef,
      progressStatus: progressStatusRef,
      buttonType: buttonTypeRef,
      showProgress: showProgressRef,
      disabled: NUpload2.mergedDisabledRef,
      showCancelButton: showCancelButtonRef,
      showRemoveButton: showRemoveButtonRef,
      showDownloadButton: showDownloadButtonRef,
      showRetryButton: showRetryButtonRef,
      showPreviewButton: showPreviewButtonRef,
      mergedThumbnailUrl: mergedThumbnailUrlRef,
      shouldUseThumbnailUrl: NUpload2.shouldUseThumbnailUrlRef,
      renderIcon: NUpload2.renderIconRef,
      imageRef,
      handleRemoveOrCancelClick,
      handleDownloadClick,
      handleRetryClick,
      handlePreviewClick
    };
  },
  render() {
    const { clsPrefix, mergedTheme, listType, file, renderIcon } = this;
    let icon;
    const isImageType = listType === "image";
    const isImageCardType = listType === "image-card";
    if (isImageType || isImageCardType) {
      icon = !this.shouldUseThumbnailUrl(file) || !this.mergedThumbnailUrl ? h("span", { class: `${clsPrefix}-upload-file-info__thumbnail` }, renderIcon ? renderIcon(file) : isImageFile(file) ? h(NBaseIcon, { clsPrefix }, { default: () => imageIcon }) : h(NBaseIcon, { clsPrefix }, { default: () => documentIcon })) : h("a", { rel: "noopener noreferer", target: "_blank", href: file.url || void 0, class: `${clsPrefix}-upload-file-info__thumbnail`, onClick: this.handlePreviewClick }, listType === "image-card" ? h(NImage, { src: this.mergedThumbnailUrl || void 0, previewSrc: file.url || void 0, alt: file.name, ref: "imageRef" }) : h("img", { src: this.mergedThumbnailUrl || void 0, alt: file.name }));
    } else {
      icon = h("span", { class: `${clsPrefix}-upload-file-info__thumbnail` }, renderIcon ? renderIcon(file) : h(NBaseIcon, { clsPrefix }, { default: () => h(AttachIcon, null) }));
    }
    const progress = h(NUploadProgress, { show: this.showProgress, percentage: file.percentage || 0, status: this.progressStatus });
    const showName = listType === "text" || listType === "image";
    return h(
      "div",
      { class: [
        `${clsPrefix}-upload-file`,
        `${clsPrefix}-upload-file--${this.progressStatus}-status`,
        file.url && file.status !== "error" && listType !== "image-card" && `${clsPrefix}-upload-file--with-url`,
        `${clsPrefix}-upload-file--${listType}-type`
      ] },
      h(
        "div",
        { class: `${clsPrefix}-upload-file-info` },
        icon,
        h(
          "div",
          { class: `${clsPrefix}-upload-file-info__name` },
          showName && (file.url && file.status !== "error" ? h("a", { rel: "noopener noreferer", target: "_blank", href: file.url || void 0, onClick: this.handlePreviewClick }, file.name) : h("span", { onClick: this.handlePreviewClick }, file.name)),
          isImageType && progress
        ),
        h(
          "div",
          { class: [
            `${clsPrefix}-upload-file-info__action`,
            `${clsPrefix}-upload-file-info__action--${listType}-type`
          ] },
          this.showPreviewButton ? h(NButton, { key: "preview", quaternary: true, type: this.buttonType, onClick: this.handlePreviewClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(EyeIcon, null) })
          }) : null,
          (this.showRemoveButton || this.showCancelButton) && !this.disabled && h(NButton, { key: "cancelOrTrash", theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, quaternary: true, builtinThemeOverrides: buttonThemeOverrides, type: this.buttonType, onClick: this.handleRemoveOrCancelClick }, {
            icon: () => h(NIconSwitchTransition, null, {
              default: () => this.showRemoveButton ? h(NBaseIcon, { clsPrefix, key: "trash" }, { default: () => h(TrashIcon, null) }) : h(NBaseIcon, { clsPrefix, key: "cancel" }, { default: () => h(CancelIcon, null) })
            })
          }),
          this.showRetryButton && !this.disabled && h(NButton, { key: "retry", quaternary: true, type: this.buttonType, onClick: this.handleRetryClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(RetryIcon, null) })
          }),
          this.showDownloadButton ? h(NButton, { key: "download", quaternary: true, type: this.buttonType, onClick: this.handleDownloadClick, theme: mergedTheme.peers.Button, themeOverrides: mergedTheme.peerOverrides.Button, builtinThemeOverrides: buttonThemeOverrides }, {
            icon: () => h(NBaseIcon, { clsPrefix }, { default: () => h(DownloadIcon, null) })
          }) : null
        )
      ),
      !isImageType && progress
    );
  }
});
const NUploadFileList = defineComponent({
  name: "UploadFileList",
  setup(_, { slots }) {
    const NUpload2 = inject(uploadInjectionKey, null);
    if (!NUpload2) {
      throwError("upload-file-list", "`n-upload-file-list` must be placed inside `n-upload`.");
    }
    const { abstractRef, mergedClsPrefixRef, listTypeRef, mergedFileListRef, fileListStyleRef, cssVarsRef, themeClassRef, maxReachedRef, showTriggerRef, imageGroupPropsRef } = NUpload2;
    const isImageCardTypeRef = computed(() => listTypeRef.value === "image-card");
    const renderFileList = () => mergedFileListRef.value.map((file) => h(NUploadFile, { clsPrefix: mergedClsPrefixRef.value, key: file.id, file, listType: listTypeRef.value }));
    const renderUploadFileList = () => isImageCardTypeRef.value ? h(NImageGroup, Object.assign({}, imageGroupPropsRef.value), { default: renderFileList }) : h(NFadeInExpandTransition, { group: true }, {
      default: renderFileList
    });
    return () => {
      const { value: mergedClsPrefix } = mergedClsPrefixRef;
      const { value: abstract } = abstractRef;
      return h(
        "div",
        { class: [
          `${mergedClsPrefix}-upload-file-list`,
          isImageCardTypeRef.value && `${mergedClsPrefix}-upload-file-list--grid`,
          abstract ? themeClassRef === null || themeClassRef === void 0 ? void 0 : themeClassRef.value : void 0
        ], style: [
          abstract && cssVarsRef ? cssVarsRef.value : "",
          fileListStyleRef.value
        ] },
        renderUploadFileList(),
        showTriggerRef.value && !maxReachedRef.value && isImageCardTypeRef.value && h(NUploadTrigger, null, slots)
      );
    };
  }
});
const style = c([cB("upload", "width: 100%;", [cM("dragger-inside", [cB("upload-trigger", `
 display: block;
 `)]), cM("drag-over", [cB("upload-dragger", `
 border: var(--n-dragger-border-hover);
 `)])]), cB("upload-dragger", `
 cursor: pointer;
 box-sizing: border-box;
 width: 100%;
 text-align: center;
 border-radius: var(--n-border-radius);
 padding: 24px;
 opacity: 1;
 transition:
 opacity .3s var(--n-bezier),
 border-color .3s var(--n-bezier),
 background-color .3s var(--n-bezier);
 background-color: var(--n-dragger-color);
 border: var(--n-dragger-border);
 `, [c("&:hover", `
 border: var(--n-dragger-border-hover);
 `), cM("disabled", `
 cursor: not-allowed;
 `)]), cB("upload-trigger", `
 display: inline-block;
 box-sizing: border-box;
 opacity: 1;
 transition: opacity .3s var(--n-bezier);
 `, [c("+", [cB("upload-file-list", "margin-top: 8px;")]), cM("disabled", `
 opacity: var(--n-item-disabled-opacity);
 cursor: not-allowed;
 `), cM("image-card", `
 width: 96px;
 height: 96px;
 `, [cB("base-icon", `
 font-size: 24px;
 `), cB("upload-dragger", `
 padding: 0;
 height: 100%;
 width: 100%;
 display: flex;
 align-items: center;
 justify-content: center;
 `)])]), cB("upload-file-list", `
 line-height: var(--n-line-height);
 opacity: 1;
 transition: opacity .3s var(--n-bezier);
 `, [c("a, img", "outline: none;"), cM("disabled", `
 opacity: var(--n-item-disabled-opacity);
 cursor: not-allowed;
 `, [cB("upload-file", "cursor: not-allowed;")]), cM("grid", `
 display: grid;
 grid-template-columns: repeat(auto-fill, 96px);
 grid-gap: 8px;
 margin-top: 0;
 `), cB("upload-file", `
 display: block;
 box-sizing: border-box;
 cursor: default;
 padding: 0px 12px 0 6px;
 transition: background-color .3s var(--n-bezier);
 border-radius: var(--n-border-radius);
 `, [fadeInHeightExpandTransition(), cB("progress", [fadeInHeightExpandTransition({
  foldPadding: true
})]), c("&:hover", `
 background-color: var(--n-item-color-hover);
 `, [cB("upload-file-info", [cE("action", `
 opacity: 1;
 `)])]), cM("image-type", `
 border-radius: var(--n-border-radius);
 text-decoration: underline;
 text-decoration-color: #0000;
 `, [cB("upload-file-info", `
 padding-top: 0px;
 padding-bottom: 0px;
 width: 100%;
 height: 100%;
 display: flex;
 justify-content: space-between;
 align-items: center;
 padding: 6px 0;
 `, [cB("progress", `
 padding: 2px 0;
 margin-bottom: 0;
 `), cE("name", `
 padding: 0 8px;
 `), cE("thumbnail", `
 width: 32px;
 height: 32px;
 font-size: 28px;
 display: flex;
 justify-content: center;
 align-items: center;
 `, [c("img", `
 width: 100%;
 `)])])]), cM("text-type", [cB("progress", `
 box-sizing: border-box;
 padding-bottom: 6px;
 margin-bottom: 6px;
 `)]), cM("image-card-type", `
 position: relative;
 width: 96px;
 height: 96px;
 border: var(--n-item-border-image-card);
 border-radius: var(--n-border-radius);
 padding: 0;
 display: flex;
 align-items: center;
 justify-content: center;
 transition: border-color .3s var(--n-bezier), background-color .3s var(--n-bezier);
 border-radius: var(--n-border-radius);
 overflow: hidden;
 `, [cB("progress", `
 position: absolute;
 left: 8px;
 bottom: 8px;
 right: 8px;
 width: unset;
 `), cB("upload-file-info", `
 padding: 0;
 width: 100%;
 height: 100%;
 `, [cE("thumbnail", `
 width: 100%;
 height: 100%;
 display: flex;
 flex-direction: column;
 align-items: center;
 justify-content: center;
 font-size: 36px;
 `, [c("img", `
 width: 100%;
 `)])]), c("&::before", `
 position: absolute;
 z-index: 1;
 left: 0;
 right: 0;
 top: 0;
 bottom: 0;
 border-radius: inherit;
 opacity: 0;
 transition: opacity .2s var(--n-bezier);
 content: "";
 `), c("&:hover", [c("&::before", "opacity: 1;"), cB("upload-file-info", [cE("thumbnail", "opacity: .12;")])])]), cM("error-status", [c("&:hover", `
 background-color: var(--n-item-color-hover-error);
 `), cB("upload-file-info", [cE("name", "color: var(--n-item-text-color-error);"), cE("thumbnail", "color: var(--n-item-text-color-error);")]), cM("image-card-type", `
 border: var(--n-item-border-image-card-error);
 `)]), cM("with-url", `
 cursor: pointer;
 `, [cB("upload-file-info", [cE("name", `
 color: var(--n-item-text-color-success);
 text-decoration-color: var(--n-item-text-color-success);
 `, [c("a", `
 text-decoration: underline;
 `)])])]), cB("upload-file-info", `
 position: relative;
 padding-top: 6px;
 padding-bottom: 6px;
 display: flex;
 flex-wrap: nowrap;
 `, [cE("thumbnail", `
 font-size: 18px;
 opacity: 1;
 transition: opacity .2s var(--n-bezier);
 color: var(--n-item-icon-color);
 `, [cB("base-icon", `
 margin-right: 2px;
 vertical-align: middle;
 transition: color .3s var(--n-bezier);
 `)]), cE("action", `
 padding-top: inherit;
 padding-bottom: inherit;
 position: absolute;
 right: 0;
 top: 0;
 bottom: 0;
 width: 80px;
 display: flex;
 align-items: center;
 transition: opacity .2s var(--n-bezier);
 justify-content: flex-end;
 opacity: 0;
 `, [cB("button", [c("&:not(:last-child)", {
  marginRight: "4px"
}), cB("base-icon", [c("svg", [iconSwitchTransition()])])]), cM("image-type", `
 position: relative;
 max-width: 80px;
 width: auto;
 `), cM("image-card-type", `
 z-index: 2;
 position: absolute;
 width: 100%;
 height: 100%;
 left: 0;
 right: 0;
 bottom: 0;
 top: 0;
 display: flex;
 justify-content: center;
 align-items: center;
 `)]), cE("name", `
 color: var(--n-item-text-color);
 flex: 1;
 display: flex;
 justify-content: center;
 text-overflow: ellipsis;
 overflow: hidden;
 flex-direction: column;
 text-decoration-color: #0000;
 font-size: var(--n-font-size);
 transition:
 color .3s var(--n-bezier),
 text-decoration-color .3s var(--n-bezier); 
 `, [c("a", `
 color: inherit;
 text-decoration: underline;
 `)])])])]), cB("upload-file-input", `
 display: block;
 width: 0;
 height: 0;
 opacity: 0;
 `)]);
var __awaiter = globalThis && globalThis.__awaiter || function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
function createXhrHandlers(inst, file, xhr) {
  const { doChange, xhrMap } = inst;
  let percentage = 0;
  function handleXHRError(e) {
    var _a;
    let fileAfterChange = Object.assign({}, file, {
      status: "error",
      percentage
    });
    xhrMap.delete(file.id);
    fileAfterChange = createSettledFileInfo(((_a = inst.onError) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange, event: e })) || fileAfterChange);
    doChange(fileAfterChange, e);
  }
  function handleXHRLoad(e) {
    var _a;
    if (inst.isErrorState) {
      if (inst.isErrorState(xhr)) {
        handleXHRError(e);
        return;
      }
    } else {
      if (xhr.status < 200 || xhr.status >= 300) {
        handleXHRError(e);
        return;
      }
    }
    let fileAfterChange = Object.assign({}, file, {
      status: "finished",
      percentage
    });
    xhrMap.delete(file.id);
    fileAfterChange = createSettledFileInfo(((_a = inst.onFinish) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange, event: e })) || fileAfterChange);
    doChange(fileAfterChange, e);
  }
  return {
    handleXHRLoad,
    handleXHRError,
    handleXHRAbort(e) {
      const fileAfterChange = Object.assign({}, file, {
        status: "removed",
        file: null,
        percentage
      });
      xhrMap.delete(file.id);
      doChange(fileAfterChange, e);
    },
    handleXHRProgress(e) {
      const fileAfterChange = Object.assign({}, file, {
        status: "uploading"
      });
      if (e.lengthComputable) {
        const progress = Math.ceil(e.loaded / e.total * 100);
        fileAfterChange.percentage = progress;
        percentage = progress;
      }
      doChange(fileAfterChange, e);
    }
  };
}
function customSubmitImpl(options) {
  const { inst, file, data, headers, withCredentials, action, customRequest } = options;
  const { doChange } = options.inst;
  let percentage = 0;
  customRequest({
    file,
    data,
    headers,
    withCredentials,
    action,
    onProgress(event) {
      const fileAfterChange = Object.assign({}, file, {
        status: "uploading"
      });
      const progress = event.percent;
      fileAfterChange.percentage = progress;
      percentage = progress;
      doChange(fileAfterChange);
    },
    onFinish() {
      var _a;
      let fileAfterChange = Object.assign({}, file, {
        status: "finished",
        percentage
      });
      fileAfterChange = createSettledFileInfo(((_a = inst.onFinish) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange })) || fileAfterChange);
      doChange(fileAfterChange);
    },
    onError() {
      var _a;
      let fileAfterChange = Object.assign({}, file, {
        status: "error",
        percentage
      });
      fileAfterChange = createSettledFileInfo(((_a = inst.onError) === null || _a === void 0 ? void 0 : _a.call(inst, { file: fileAfterChange })) || fileAfterChange);
      doChange(fileAfterChange);
    }
  });
}
function registerHandler(inst, file, request) {
  const handlers = createXhrHandlers(inst, file, request);
  request.onabort = handlers.handleXHRAbort;
  request.onerror = handlers.handleXHRError;
  request.onload = handlers.handleXHRLoad;
  if (request.upload) {
    request.upload.onprogress = handlers.handleXHRProgress;
  }
}
function unwrapFunctionValue(data, file) {
  if (typeof data === "function") {
    return data({ file });
  }
  if (data)
    return data;
  return {};
}
function setHeaders(request, headers, file) {
  const headersObject = unwrapFunctionValue(headers, file);
  if (!headersObject)
    return;
  Object.keys(headersObject).forEach((key) => {
    request.setRequestHeader(key, headersObject[key]);
  });
}
function appendData(formData, data, file) {
  const dataObject = unwrapFunctionValue(data, file);
  if (!dataObject)
    return;
  Object.keys(dataObject).forEach((key) => {
    formData.append(key, dataObject[key]);
  });
}
function submitImpl(inst, fieldName, file, { method, action, withCredentials, responseType, headers, data }) {
  const request = new XMLHttpRequest();
  request.responseType = responseType;
  inst.xhrMap.set(file.id, request);
  request.withCredentials = withCredentials;
  const formData = new FormData();
  appendData(formData, data, file);
  formData.append(fieldName, file.file);
  registerHandler(inst, file, request);
  if (action !== void 0) {
    request.open(method.toUpperCase(), action);
    setHeaders(request, headers, file);
    request.send(formData);
    const fileAfterChange = Object.assign({}, file, {
      status: "uploading"
    });
    inst.doChange(fileAfterChange);
  }
}
const uploadProps = Object.assign(Object.assign({}, useTheme.props), {
  name: {
    type: String,
    default: "file"
  },
  accept: String,
  action: String,
  customRequest: Function,
  directory: Boolean,
  directoryDnd: { type: Boolean, default: void 0 },
  method: {
    type: String,
    default: "POST"
  },
  multiple: Boolean,
  showFileList: {
    type: Boolean,
    default: true
  },
  data: [Object, Function],
  headers: [Object, Function],
  withCredentials: Boolean,
  responseType: {
    type: String,
    default: ""
  },
  disabled: {
    type: Boolean,
    default: void 0
  },
  onChange: Function,
  onRemove: Function,
  onFinish: Function,
  onError: Function,
  onBeforeUpload: Function,
  isErrorState: Function,
  /** currently not used */
  onDownload: Function,
  defaultUpload: {
    type: Boolean,
    default: true
  },
  fileList: Array,
  "onUpdate:fileList": [Function, Array],
  onUpdateFileList: [Function, Array],
  fileListStyle: [String, Object],
  defaultFileList: {
    type: Array,
    default: () => []
  },
  showCancelButton: {
    type: Boolean,
    default: true
  },
  showRemoveButton: {
    type: Boolean,
    default: true
  },
  showDownloadButton: Boolean,
  showRetryButton: {
    type: Boolean,
    default: true
  },
  showPreviewButton: {
    type: Boolean,
    default: true
  },
  listType: {
    type: String,
    default: "text"
  },
  onPreview: Function,
  shouldUseThumbnailUrl: {
    type: Function,
    default: (file) => {
      if (!environmentSupportFile)
        return false;
      return isImageFile(file);
    }
  },
  createThumbnailUrl: Function,
  abstract: Boolean,
  max: Number,
  showTrigger: {
    type: Boolean,
    default: true
  },
  imageGroupProps: Object,
  inputProps: Object,
  triggerStyle: [String, Object],
  renderIcon: Object
});
const NUpload = defineComponent({
  name: "Upload",
  props: uploadProps,
  setup(props) {
    if (props.abstract && props.listType === "image-card") {
      throwError("upload", "when the list-type is image-card, abstract is not supported.");
    }
    const { mergedClsPrefixRef, inlineThemeDisabled } = useConfig(props);
    const themeRef = useTheme("Upload", "-upload", style, uploadLight, props, mergedClsPrefixRef);
    const formItem = useFormItem(props);
    const maxReachedRef = computed(() => {
      const { max } = props;
      if (max !== void 0) {
        return mergedFileListRef.value.length >= max;
      }
      return false;
    });
    const uncontrolledFileListRef = ref(props.defaultFileList);
    const controlledFileListRef = toRef(props, "fileList");
    const inputElRef = ref(null);
    const draggerInsideRef = {
      value: false
    };
    const dragOverRef = ref(false);
    const xhrMap = /* @__PURE__ */ new Map();
    const _mergedFileListRef = useMergedState(controlledFileListRef, uncontrolledFileListRef);
    const mergedFileListRef = computed(() => _mergedFileListRef.value.map(createSettledFileInfo));
    function openOpenFileDialog() {
      var _a;
      (_a = inputElRef.value) === null || _a === void 0 ? void 0 : _a.click();
    }
    function handleFileInputChange(e) {
      const target = e.target;
      handleFileAddition(target.files ? Array.from(target.files).map((file) => ({
        file,
        entry: null,
        source: "input"
      })) : null, e);
      target.value = "";
    }
    function doUpdateFileList(files) {
      const { "onUpdate:fileList": _onUpdateFileList, onUpdateFileList } = props;
      if (_onUpdateFileList)
        call(_onUpdateFileList, files);
      if (onUpdateFileList)
        call(onUpdateFileList, files);
      uncontrolledFileListRef.value = files;
    }
    const mergedMultipleRef = computed(() => props.multiple || props.directory);
    function handleFileAddition(fileAndEntries, e) {
      if (!fileAndEntries || fileAndEntries.length === 0)
        return;
      const { onBeforeUpload } = props;
      fileAndEntries = mergedMultipleRef.value ? fileAndEntries : [fileAndEntries[0]];
      const { max, accept } = props;
      fileAndEntries = fileAndEntries.filter(({ file, source }) => {
        if (source === "dnd" && (accept === null || accept === void 0 ? void 0 : accept.trim())) {
          return matchType(file.name, file.type, accept);
        } else {
          return true;
        }
      });
      if (max) {
        fileAndEntries = fileAndEntries.slice(0, max - mergedFileListRef.value.length);
      }
      const batchId = createId();
      void Promise.all(fileAndEntries.map(({ file, entry }) => __awaiter(this, void 0, void 0, function* () {
        var _a;
        const fileInfo = {
          id: createId(),
          batchId,
          name: file.name,
          status: "pending",
          percentage: 0,
          file,
          url: null,
          type: file.type,
          thumbnailUrl: null,
          fullPath: (_a = entry === null || entry === void 0 ? void 0 : entry.fullPath) !== null && _a !== void 0 ? _a : `/${file.webkitRelativePath || file.name}`
        };
        if (!onBeforeUpload || (yield onBeforeUpload({
          file: fileInfo,
          fileList: mergedFileListRef.value
        })) !== false) {
          return fileInfo;
        }
        return null;
      }))).then((fileInfos) => __awaiter(this, void 0, void 0, function* () {
        let nextTickChain = Promise.resolve();
        fileInfos.forEach((fileInfo) => {
          nextTickChain = nextTickChain.then(nextTick).then(() => {
            fileInfo && doChange(fileInfo, e, {
              append: true
            });
          });
        });
        yield nextTickChain;
      })).then(() => {
        if (props.defaultUpload) {
          submit();
        }
      });
    }
    function submit(fileId) {
      const { method, action, withCredentials, headers, data, name: fieldName } = props;
      const filesToUpload = fileId !== void 0 ? mergedFileListRef.value.filter((file) => file.id === fileId) : mergedFileListRef.value;
      const shouldReupload = fileId !== void 0;
      filesToUpload.forEach((file) => {
        const { status } = file;
        if (status === "pending" || status === "error" && shouldReupload) {
          if (props.customRequest) {
            customSubmitImpl({
              inst: {
                doChange,
                xhrMap,
                onFinish: props.onFinish,
                onError: props.onError
              },
              file,
              action,
              withCredentials,
              headers,
              data,
              customRequest: props.customRequest
            });
          } else {
            submitImpl({
              doChange,
              xhrMap,
              onFinish: props.onFinish,
              onError: props.onError,
              isErrorState: props.isErrorState
            }, fieldName, file, {
              method,
              action,
              withCredentials,
              responseType: props.responseType,
              headers,
              data
            });
          }
        }
      });
    }
    const doChange = (fileAfterChange, event, options = {
      append: false,
      remove: false
    }) => {
      const { append, remove } = options;
      const fileListAfterChange = Array.from(mergedFileListRef.value);
      const fileIndex = fileListAfterChange.findIndex((file) => file.id === fileAfterChange.id);
      if (append || remove || ~fileIndex) {
        if (append) {
          fileListAfterChange.push(fileAfterChange);
        } else if (remove) {
          fileListAfterChange.splice(fileIndex, 1);
        } else {
          fileListAfterChange.splice(fileIndex, 1, fileAfterChange);
        }
        const { onChange } = props;
        if (onChange) {
          onChange({
            file: fileAfterChange,
            fileList: fileListAfterChange,
            event
          });
        }
        doUpdateFileList(fileListAfterChange);
      }
    };
    function getFileThumbnailUrlResolver(file) {
      var _a;
      if (file.thumbnailUrl)
        return file.thumbnailUrl;
      const { createThumbnailUrl } = props;
      if (createThumbnailUrl) {
        return (_a = createThumbnailUrl(file.file, file)) !== null && _a !== void 0 ? _a : file.url || "";
      }
      if (file.url) {
        return file.url;
      } else if (file.file) {
        return createImageDataUrl(file.file);
      }
      return "";
    }
    const cssVarsRef = computed(() => {
      const { common: { cubicBezierEaseInOut }, self: { draggerColor, draggerBorder, draggerBorderHover, itemColorHover, itemColorHoverError, itemTextColorError, itemTextColorSuccess, itemTextColor, itemIconColor, itemDisabledOpacity, lineHeight, borderRadius, fontSize, itemBorderImageCardError, itemBorderImageCard } } = themeRef.value;
      return {
        "--n-bezier": cubicBezierEaseInOut,
        "--n-border-radius": borderRadius,
        "--n-dragger-border": draggerBorder,
        "--n-dragger-border-hover": draggerBorderHover,
        "--n-dragger-color": draggerColor,
        "--n-font-size": fontSize,
        "--n-item-color-hover": itemColorHover,
        "--n-item-color-hover-error": itemColorHoverError,
        "--n-item-disabled-opacity": itemDisabledOpacity,
        "--n-item-icon-color": itemIconColor,
        "--n-item-text-color": itemTextColor,
        "--n-item-text-color-error": itemTextColorError,
        "--n-item-text-color-success": itemTextColorSuccess,
        "--n-line-height": lineHeight,
        "--n-item-border-image-card-error": itemBorderImageCardError,
        "--n-item-border-image-card": itemBorderImageCard
      };
    });
    const themeClassHandle = inlineThemeDisabled ? useThemeClass("upload", void 0, cssVarsRef, props) : void 0;
    provide(uploadInjectionKey, {
      mergedClsPrefixRef,
      mergedThemeRef: themeRef,
      showCancelButtonRef: toRef(props, "showCancelButton"),
      showDownloadButtonRef: toRef(props, "showDownloadButton"),
      showRemoveButtonRef: toRef(props, "showRemoveButton"),
      showRetryButtonRef: toRef(props, "showRetryButton"),
      onRemoveRef: toRef(props, "onRemove"),
      onDownloadRef: toRef(props, "onDownload"),
      mergedFileListRef,
      triggerStyleRef: toRef(props, "triggerStyle"),
      shouldUseThumbnailUrlRef: toRef(props, "shouldUseThumbnailUrl"),
      renderIconRef: toRef(props, "renderIcon"),
      xhrMap,
      submit,
      doChange,
      showPreviewButtonRef: toRef(props, "showPreviewButton"),
      onPreviewRef: toRef(props, "onPreview"),
      getFileThumbnailUrlResolver,
      listTypeRef: toRef(props, "listType"),
      dragOverRef,
      openOpenFileDialog,
      draggerInsideRef,
      handleFileAddition,
      mergedDisabledRef: formItem.mergedDisabledRef,
      maxReachedRef,
      fileListStyleRef: toRef(props, "fileListStyle"),
      abstractRef: toRef(props, "abstract"),
      acceptRef: toRef(props, "accept"),
      cssVarsRef: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClassRef: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender,
      showTriggerRef: toRef(props, "showTrigger"),
      imageGroupPropsRef: toRef(props, "imageGroupProps"),
      mergedDirectoryDndRef: computed(() => {
        var _a;
        return (_a = props.directoryDnd) !== null && _a !== void 0 ? _a : props.directory;
      })
    });
    const exposedMethods = {
      clear: () => {
        uncontrolledFileListRef.value = [];
      },
      submit,
      openOpenFileDialog
    };
    return Object.assign({
      mergedClsPrefix: mergedClsPrefixRef,
      draggerInsideRef,
      inputElRef,
      mergedTheme: themeRef,
      dragOver: dragOverRef,
      mergedMultiple: mergedMultipleRef,
      cssVars: inlineThemeDisabled ? void 0 : cssVarsRef,
      themeClass: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.themeClass,
      onRender: themeClassHandle === null || themeClassHandle === void 0 ? void 0 : themeClassHandle.onRender,
      handleFileInputChange
    }, exposedMethods);
  },
  render() {
    var _a, _b;
    const { draggerInsideRef, mergedClsPrefix, $slots, directory, onRender } = this;
    if ($slots.default && !this.abstract) {
      const firstChild = $slots.default()[0];
      if ((_a = firstChild === null || firstChild === void 0 ? void 0 : firstChild.type) === null || _a === void 0 ? void 0 : _a[uploadDraggerKey]) {
        draggerInsideRef.value = true;
      }
    }
    const inputNode = h("input", Object.assign({}, this.inputProps, {
      ref: "inputElRef",
      type: "file",
      class: `${mergedClsPrefix}-upload-file-input`,
      accept: this.accept,
      multiple: this.mergedMultiple,
      onChange: this.handleFileInputChange,
      // @ts-expect-error // seems vue-tsc will add the prop, so we can't use expect-error
      webkitdirectory: directory || void 0,
      directory: directory || void 0
    }));
    if (this.abstract) {
      return h(
        Fragment,
        null,
        (_b = $slots.default) === null || _b === void 0 ? void 0 : _b.call($slots),
        h(Teleport, { to: "body" }, inputNode)
      );
    }
    onRender === null || onRender === void 0 ? void 0 : onRender();
    return h(
      "div",
      { class: [
        `${mergedClsPrefix}-upload`,
        draggerInsideRef.value && `${mergedClsPrefix}-upload--dragger-inside`,
        this.dragOver && `${mergedClsPrefix}-upload--drag-over`,
        this.themeClass
      ], style: this.cssVars },
      inputNode,
      this.showTrigger && this.listType !== "image-card" && h(NUploadTrigger, null, $slots),
      this.showFileList && h(NUploadFileList, null, $slots)
    );
  }
});
const _hoisted_1$6 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$6 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M393.87 190a32.1 32.1 0 0 1-45.25 0l-26.57-26.57a32.09 32.09 0 0 1 0-45.26L382.19 58a1 1 0 0 0-.3-1.64c-38.82-16.64-89.15-8.16-121.11 23.57c-30.58 30.35-32.32 76-21.12 115.84a31.93 31.93 0 0 1-9.06 32.08L64 380a48.17 48.17 0 1 0 68 68l153.86-167a31.93 31.93 0 0 1 31.6-9.13c39.54 10.59 84.54 8.6 114.72-21.19c32.49-32 39.5-88.56 23.75-120.93a1 1 0 0 0-1.6-.26z",
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "32"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$6 = /* @__PURE__ */ createBaseVNode(
  "circle",
  {
    cx: "96",
    cy: "416",
    r: "16",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$6 = [_hoisted_2$6, _hoisted_3$6];
const BuildOutline = defineComponent({
  name: "BuildOutline",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$6, _hoisted_4$6);
  }
});
const _hoisted_1$5 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$5 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M261.56 101.28a8 8 0 0 0-11.06 0L66.4 277.15a8 8 0 0 0-2.47 5.79L63.9 448a32 32 0 0 0 32 32H192a16 16 0 0 0 16-16V328a8 8 0 0 1 8-8h80a8 8 0 0 1 8 8v136a16 16 0 0 0 16 16h96.06a32 32 0 0 0 32-32V282.94a8 8 0 0 0-2.47-5.79z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$5 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M490.91 244.15l-74.8-71.56V64a16 16 0 0 0-16-16h-48a16 16 0 0 0-16 16v32l-57.92-55.38C272.77 35.14 264.71 32 256 32c-8.68 0-16.72 3.14-22.14 8.63l-212.7 203.5c-6.22 6-7 15.87-1.34 22.37A16 16 0 0 0 43 267.56L250.5 69.28a8 8 0 0 1 11.06 0l207.52 198.28a16 16 0 0 0 22.59-.44c6.14-6.36 5.63-16.86-.76-22.97z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$5 = [_hoisted_2$5, _hoisted_3$5];
const Home = defineComponent({
  name: "Home",
  render: function render2(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$5, _hoisted_4$5);
  }
});
const _hoisted_1$4 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 152h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 256h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    stroke: "currentColor",
    "stroke-linecap": "round",
    "stroke-miterlimit": "10",
    "stroke-width": "48",
    d: "M88 360h336"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_5$2 = [_hoisted_2$4, _hoisted_3$4, _hoisted_4$4];
const Menu = defineComponent({
  name: "Menu",
  render: function render3(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$4, _hoisted_5$2);
  }
});
const _hoisted_1$3 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode(
  "circle",
  {
    cx: "256",
    cy: "256",
    r: "48",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M470.39 300l-.47-.38l-31.56-24.75a16.11 16.11 0 0 1-6.1-13.33v-11.56a16 16 0 0 1 6.11-13.22L469.92 212l.47-.38a26.68 26.68 0 0 0 5.9-34.06l-42.71-73.9a1.59 1.59 0 0 1-.13-.22A26.86 26.86 0 0 0 401 92.14l-.35.13l-37.1 14.93a15.94 15.94 0 0 1-14.47-1.29q-4.92-3.1-10-5.86a15.94 15.94 0 0 1-8.19-11.82l-5.59-39.59l-.12-.72A27.22 27.22 0 0 0 298.76 26h-85.52a26.92 26.92 0 0 0-26.45 22.39l-.09.56l-5.57 39.67a16 16 0 0 1-8.13 11.82a175.21 175.21 0 0 0-10 5.82a15.92 15.92 0 0 1-14.43 1.27l-37.13-15l-.35-.14a26.87 26.87 0 0 0-32.48 11.34l-.13.22l-42.77 73.95a26.71 26.71 0 0 0 5.9 34.1l.47.38l31.56 24.75a16.11 16.11 0 0 1 6.1 13.33v11.56a16 16 0 0 1-6.11 13.22L42.08 300l-.47.38a26.68 26.68 0 0 0-5.9 34.06l42.71 73.9a1.59 1.59 0 0 1 .13.22a26.86 26.86 0 0 0 32.45 11.3l.35-.13l37.07-14.93a15.94 15.94 0 0 1 14.47 1.29q4.92 3.11 10 5.86a15.94 15.94 0 0 1 8.19 11.82l5.56 39.59l.12.72A27.22 27.22 0 0 0 213.24 486h85.52a26.92 26.92 0 0 0 26.45-22.39l.09-.56l5.57-39.67a16 16 0 0 1 8.18-11.82c3.42-1.84 6.76-3.79 10-5.82a15.92 15.92 0 0 1 14.43-1.27l37.13 14.95l.35.14a26.85 26.85 0 0 0 32.48-11.34a2.53 2.53 0 0 1 .13-.22l42.71-73.89a26.7 26.7 0 0 0-5.89-34.11zm-134.48-40.24a80 80 0 1 1-83.66-83.67a80.21 80.21 0 0 1 83.66 83.67z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$3 = [_hoisted_2$3, _hoisted_3$3];
const Settings = defineComponent({
  name: "Settings",
  render: function render4(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$3, _hoisted_4$3);
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "WIP",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NResult), {
        title: "Work in progress",
        description: "This page is still under development.",
        style: { "height": "70vh", "display": "flex", "align-items": "center", "justify-content": "center" }
      }, {
        icon: withCtx(() => [
          createVNode(unref(NIcon), { size: "250" }, {
            default: withCtx(() => [
              createVNode(unref(BuildOutline))
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "CivitAIDownload",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$5);
    };
  }
});
const _withScopeId = (n) => (pushScopeId("data-v-6a6fb4b4"), n = n(), popScopeId(), n);
const _hoisted_1$2 = { style: { "margin": "18px" } };
const _hoisted_2$2 = { style: { "width": "100%", "display": "inline-flex", "justify-content": "space-between", "align-items": "center" } };
const _hoisted_3$2 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("div", null, "Install custom models from Hugging Face", -1));
const _hoisted_4$2 = { style: { "display": "inline-flex", "align-items": "center" } };
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "HuggingfaceDownload",
  setup(__props) {
    const conf = useState();
    const message = useMessage();
    const customModel = ref("");
    function downloadModel(model) {
      const url = new URL(`${serverUrl}/api/models/download`);
      const modelName = typeof model === "string" ? model : model.value;
      url.searchParams.append("model", modelName);
      console.log(url);
      conf.state.downloading = true;
      customModel.value = "";
      message.info(`Downloading model: ${modelName}`);
      fetch(url, { method: "POST" }).then(() => {
        conf.state.downloading = false;
        message.success(`Downloaded model: ${modelName}`);
      }).catch(() => {
        conf.state.downloading = false;
        message.error(`Failed to download model: ${modelName}`);
      });
    }
    const renderIcon = (icon, size = "medium") => {
      return () => {
        return h(
          NIcon,
          {
            size
          },
          {
            default: () => h(icon)
          }
        );
      };
    };
    function getPluginOptions(row) {
      const options = [
        {
          label: "Hugging Face",
          key: "github",
          icon: renderIcon(Home),
          props: {
            onClick: () => window.open(row.huggingface_url, "_blank")
          }
        }
      ];
      return options;
    }
    const columns = [
      {
        title: "Name",
        key: "name",
        sorter: "default"
      },
      {
        title: "Repository",
        key: "huggingface_id",
        sorter: "default"
      },
      {
        title: "Download",
        key: "download",
        render(row) {
          return h(
            NButton,
            {
              type: "primary",
              secondary: true,
              round: true,
              block: true,
              bordered: false,
              disabled: conf.state.downloading,
              onClick: () => {
                downloadModel(row.huggingface_id);
              }
            },
            { default: () => "Download" }
          );
        }
      },
      {
        title: "",
        width: 60,
        key: "menu",
        render(row) {
          return h(
            NDropdown,
            {
              trigger: "hover",
              options: getPluginOptions(row),
              disabled: conf.state.downloading
            },
            { default: renderIcon(Menu) }
          );
        },
        filter: "default"
      }
    ];
    const modelData = reactive([]);
    const modelFilter = ref("");
    const dataRef = computed(() => {
      if (modelFilter.value !== "") {
        return modelData.filter(
          (model) => model.name.toLowerCase().includes(modelFilter.value.toLowerCase())
        );
      } else {
        return modelData;
      }
    });
    const pagination = reactive({ pageSize: 10 });
    fetch(huggingfaceModelsFile).then((res) => {
      res.json().then((data) => {
        modelData.push(...data["models"]);
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$2, [
        createVNode(unref(NCard), {
          title: "Custom model",
          segmented: ""
        }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2$2, [
              _hoisted_3$2,
              createBaseVNode("div", _hoisted_4$2, [
                createVNode(unref(NInput), {
                  value: customModel.value,
                  "onUpdate:value": _cache[0] || (_cache[0] = ($event) => customModel.value = $event),
                  placeholder: "andite/anything-v4.0",
                  style: { "width": "350px" }
                }, null, 8, ["value"]),
                createVNode(unref(NButton), {
                  type: "primary",
                  bordered: "",
                  onClick: _cache[1] || (_cache[1] = ($event) => downloadModel(customModel.value)),
                  loading: unref(conf).state.downloading,
                  disabled: unref(conf).state.downloading || customModel.value === "",
                  secondary: "",
                  style: { "margin-right": "16px", "margin-left": "4px" }
                }, {
                  default: withCtx(() => [
                    createTextVNode("Install")
                  ]),
                  _: 1
                }, 8, ["loading", "disabled"])
              ])
            ])
          ]),
          _: 1
        }),
        createVNode(unref(NCard), {
          title: "Currated models",
          style: { "margin-top": "12px" },
          segmented: ""
        }, {
          default: withCtx(() => [
            createVNode(unref(NInput), {
              value: modelFilter.value,
              "onUpdate:value": _cache[2] || (_cache[2] = ($event) => modelFilter.value = $event),
              style: { "margin-bottom": "12px" },
              placeholder: "Filter",
              clearable: ""
            }, null, 8, ["value"]),
            createVNode(unref(NDataTable), {
              columns,
              data: dataRef.value,
              pagination,
              bordered: true,
              style: { "padding-bottom": "24px" }
            }, null, 8, ["data", "pagination"])
          ]),
          _: 1
        })
      ]);
    };
  }
});
const HuggingfaceDownload_vue_vue_type_style_index_0_scoped_6a6fb4b4_lang = "";
const HuggingfaceDownload = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["__scopeId", "data-v-6a6fb4b4"]]);
const _hoisted_1$1 = { style: { "margin": "16px" } };
const _hoisted_2$1 = { class: "flex-container" };
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "FP32", -1);
const _hoisted_4$1 = { class: "flex-container" };
const _hoisted_5$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Output in safetensors format", -1);
const _hoisted_6$1 = { class: "flex-container" };
const _hoisted_7$1 = /* @__PURE__ */ createBaseVNode("p", { class: "slider-label" }, "Model", -1);
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "ModelConvert",
  setup(__props) {
    const message = useMessage();
    const model = ref("");
    const modelOptions = reactive([]);
    const building = ref(false);
    const use_fp32 = ref(false);
    const safetensors = ref(false);
    const showUnloadModal = ref(false);
    fetch(`${serverUrl}/api/models/available`).then((res) => {
      res.json().then((data) => {
        modelOptions.splice(0, modelOptions.length);
        const pyTorch = data.filter((x) => x.backend === "PyTorch");
        if (pyTorch) {
          for (const model2 of pyTorch) {
            modelOptions.push({
              label: model2.name,
              value: model2.name,
              disabled: !model2.valid
            });
          }
        }
      });
    });
    const accelerateUnload = async () => {
      try {
        await fetch(`${serverUrl}/api/models/unload-all`, {
          method: "POST"
        });
        showUnloadModal.value = false;
        await convert();
      } catch {
        showUnloadModal.value = false;
        message.error("Failed to unload, check the console for more info.");
      }
    };
    const convert = async () => {
      showUnloadModal.value = false;
      building.value = true;
      await fetch(`${serverUrl}/api/generate/convert-model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: model.value,
          use_fp32: use_fp32.value,
          safetensors: safetensors.value
        })
      }).then(() => {
        building.value = false;
      }).catch(() => {
        building.value = false;
        message.error("Failed to accelerate, check the console for more info.");
      });
    };
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(NCard), { style: { "margin-top": "16px" } }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_2$1, [
              _hoisted_3$1,
              createVNode(unref(NSwitch), {
                value: use_fp32.value,
                "onUpdate:value": _cache[0] || (_cache[0] = ($event) => use_fp32.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_4$1, [
              _hoisted_5$1,
              createVNode(unref(NSwitch), {
                value: safetensors.value,
                "onUpdate:value": _cache[1] || (_cache[1] = ($event) => safetensors.value = $event)
              }, null, 8, ["value"])
            ]),
            createBaseVNode("div", _hoisted_6$1, [
              _hoisted_7$1,
              createVNode(unref(NSelect), {
                value: model.value,
                "onUpdate:value": _cache[2] || (_cache[2] = ($event) => model.value = $event),
                options: modelOptions,
                style: { "margin-right": "12px" }
              }, null, 8, ["value", "options"])
            ])
          ]),
          _: 1
        }),
        createVNode(unref(NSpace), {
          vertical: "",
          justify: "center",
          style: { "width": "100%" },
          align: "center"
        }, {
          default: withCtx(() => [
            createVNode(unref(NButton), {
              style: { "margin-top": "16px", "padding": "0 92px" },
              type: "success",
              ghost: "",
              loading: building.value,
              disabled: building.value || modelOptions.length === 0,
              onClick: _cache[3] || (_cache[3] = ($event) => showUnloadModal.value = true)
            }, {
              default: withCtx(() => [
                createTextVNode("Convert")
              ]),
              _: 1
            }, 8, ["loading", "disabled"])
          ]),
          _: 1
        }),
        createVNode(unref(NModal), {
          show: showUnloadModal.value,
          "onUpdate:show": _cache[4] || (_cache[4] = ($event) => showUnloadModal.value = $event),
          preset: "dialog",
          title: "Unload other models",
          width: "400px",
          closable: false,
          "show-close": false,
          content: "Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?",
          "positive-text": "Unload models",
          "negative-text": "Keep models",
          onPositiveClick: accelerateUnload,
          onNegativeClick: convert
        }, null, 8, ["show"])
      ]);
    };
  }
});
const _hoisted_1 = { style: { "padding": "12px" } };
const _hoisted_2 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_3 = { style: { "display": "inline-flex" } };
const _hoisted_4 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_5 = { style: { "display": "inline-flex" } };
const _hoisted_6 = { style: { "margin-bottom": "12px", "display": "block" } };
const _hoisted_7 = { style: { "display": "inline-flex" } };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "ModelManager",
  setup(__props) {
    const global = useState();
    const filter = ref("");
    const message = useMessage();
    const renderIcon = (icon) => {
      return () => {
        return h(NIcon, null, {
          default: () => h(icon)
        });
      };
    };
    const filteredModels = computed(() => {
      return global.state.models.filter((model) => {
        return model.path.toLowerCase().includes(filter.value.toLowerCase()) || filter.value === "";
      }).sort((a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1);
    });
    const pyTorchModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "PyTorch" && model.valid === true;
      }).sort((a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1);
    });
    const loraModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "LoRA";
      }).sort((a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1);
    });
    const textualInversionModels = computed(() => {
      return filteredModels.value.filter((model) => {
        return model.backend === "Textual Inversion";
      }).sort((a, b) => a.name.toLowerCase() < b.name.toLowerCase() ? -1 : 1);
    });
    function createPyTorchOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
        // {
        //   label: "Convert",
        //   key: `convert:${model_path}`,
        //   icon: renderIcon(GitCompare),
        // },
        // {
        //   label: "Accelerate",
        //   key: `accelerate:${model_path}`,
        //   icon: renderIcon(PlayForward),
        // },
      ];
    }
    function createLoraOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
      ];
    }
    function createTextualInversionOptions(model_path) {
      return [
        {
          label: "Delete",
          key: `delete:${model_path}`,
          icon: renderIcon(TrashBin)
        }
      ];
    }
    function deleteModel(model_path, model_type) {
      fetch(`${serverUrl}/api/models/delete-model`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model_path,
          model_type
        })
      }).then((response) => response.json()).then(() => {
        message.success("Model deleted");
      }).catch((error) => {
        message.error(error);
      });
    }
    function handlePyTorchModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "pytorch");
      } else if (action === "convert") {
        message.success(key);
      } else if (action === "accelerate") {
        message.success(key);
      }
    }
    function handleLoraModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "lora");
      }
    }
    function handleTextualInversionModelAction(key) {
      const [action, model_path] = key.split(":");
      if (action === "delete") {
        deleteModel(model_path, "textual-inversion");
      }
    }
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(NInput), {
          value: filter.value,
          "onUpdate:value": _cache[0] || (_cache[0] = ($event) => filter.value = $event),
          style: { "width": "100%", "margin-bottom": "12px" },
          placeholder: "Filter",
          clearable: ""
        }, null, 8, ["value"]),
        createVNode(unref(NGrid), {
          cols: "3",
          "x-gap": "12"
        }, {
          default: withCtx(() => [
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Model" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model`,
                      max: 5,
                      accept: ".ckpt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_2, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Model ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(pyTorchModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_3, [
                          createVNode(unref(NDropdown), {
                            options: createPyTorchOptions(model.path),
                            placement: "right",
                            onSelect: handlePyTorchModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "LoRA" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model?type=lora`,
                      max: 5,
                      accept: ".ckpt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_4, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" LoRA ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(loraModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_5, [
                          createVNode(unref(NDropdown), {
                            options: createLoraOptions(model.path),
                            placement: "right",
                            onSelect: handleLoraModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            }),
            createVNode(unref(NGi), null, {
              default: withCtx(() => [
                createVNode(unref(NCard), { title: "Textual Inversion" }, {
                  default: withCtx(() => [
                    createVNode(unref(NUpload), {
                      multiple: "",
                      "directory-dnd": "",
                      action: `${unref(serverUrl)}/api/models/upload-model?type=textual-inversion`,
                      max: 5,
                      accept: ".pt,.safetensors",
                      style: { "border-bottom": "1px solid rgb(66, 66, 71)", "padding-bottom": "12px" }
                    }, {
                      default: withCtx(() => [
                        createVNode(unref(NUploadDragger), { style: { "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center" } }, {
                          default: withCtx(() => [
                            createBaseVNode("div", _hoisted_6, [
                              createVNode(unref(NIcon), {
                                size: "48",
                                depth: 3
                              }, {
                                default: withCtx(() => [
                                  createVNode(unref(CloudUpload))
                                ]),
                                _: 1
                              })
                            ]),
                            createVNode(unref(NText), { style: { "font-size": "24px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Textual Inversion ")
                              ]),
                              _: 1
                            }),
                            createVNode(unref(NText), { style: { "font-size": "16px" } }, {
                              default: withCtx(() => [
                                createTextVNode(" Click or drag a model to this area to upload it to the server ")
                              ]),
                              _: 1
                            })
                          ]),
                          _: 1
                        })
                      ]),
                      _: 1
                    }, 8, ["action"]),
                    (openBlock(true), createElementBlock(Fragment, null, renderList(textualInversionModels.value, (model) => {
                      return openBlock(), createElementBlock("div", {
                        style: { "display": "inline-flex", "width": "100%", "align-items": "center", "justify-content": "space-between", "border-bottom": "1px solid rgb(66, 66, 71)" },
                        key: model.path
                      }, [
                        createBaseVNode("p", null, toDisplayString(model.name), 1),
                        createBaseVNode("div", _hoisted_7, [
                          createVNode(unref(NDropdown), {
                            options: createTextualInversionOptions(model.path),
                            placement: "right",
                            onSelect: handleTextualInversionModelAction
                          }, {
                            default: withCtx(() => [
                              createVNode(unref(NButton), {
                                "render-icon": renderIcon(unref(Settings))
                              }, null, 8, ["render-icon"])
                            ]),
                            _: 2
                          }, 1032, ["options"])
                        ])
                      ]);
                    }), 128))
                  ]),
                  _: 1
                })
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]);
    };
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ModelsView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(NTabs), { type: "segment" }, {
        default: withCtx(() => [
          createVNode(unref(NTabPane), { name: "Manager" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$1)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Huggingface" }, {
            default: withCtx(() => [
              createVNode(HuggingfaceDownload)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "CivitAI" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$4)
            ]),
            _: 1
          }),
          createVNode(unref(NTabPane), { name: "Convert" }, {
            default: withCtx(() => [
              createVNode(_sfc_main$2)
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as default
};
