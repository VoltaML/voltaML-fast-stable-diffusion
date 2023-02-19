import { d as defineComponent, o as openBlock, j as createElementBlock, a as createBaseVNode, G as h } from "./index.js";
const _hoisted_1$3 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M48 399.26C48 335.19 62.44 284 90.91 247c34.38-44.67 88.68-68.77 161.56-71.75V72L464 252L252.47 432V329.35c-44.25 1.19-77.66 7.58-104.27 19.84c-28.75 13.25-49.6 33.05-72.08 58.7L48 440z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$3 = [_hoisted_2$3];
const ArrowRedoSharp = defineComponent({
  name: "ArrowRedoSharp",
  render: function render(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$3, _hoisted_3$3);
  }
});
const _hoisted_1$2 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M464 440l-28.12-32.11c-22.48-25.65-43.33-45.45-72.08-58.7c-26.61-12.26-60-18.65-104.27-19.84V432L48 252L259.53 72v103.21c72.88 3 127.18 27.08 161.56 71.75C449.56 284 464 335.19 464 399.26z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$2 = [_hoisted_2$2];
const ArrowUndoSharp = defineComponent({
  name: "ArrowUndoSharp",
  render: function render2(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$2, _hoisted_3$2);
  }
});
const _hoisted_1$1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M480 96l-64-64l-244 260l64 64z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3$1 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M142 320c-36.52 0-66 30.63-66 68.57c0 25.43-31 45.72-44 45.72C52.24 462.17 86.78 480 120 480c48.62 0 88-40.91 88-91.43c0-37.94-29.48-68.57-66-68.57z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4$1 = [_hoisted_2$1, _hoisted_3$1];
const BrushSharp = defineComponent({
  name: "BrushSharp",
  render: function render3(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1$1, _hoisted_4$1);
  }
});
const _hoisted_1 = {
  xmlns: "http://www.w3.org/2000/svg",
  "xmlns:xlink": "http://www.w3.org/1999/xlink",
  viewBox: "0 0 512 512"
};
const _hoisted_2 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    d: "M337.46 240L312 214.54l-56 56l-56-56L174.54 240l56 56l-56 56L200 377.46l56-56l56 56L337.46 352l-56-56l56-56z"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    fill: "none",
    d: "M337.46 240L312 214.54l-56 56l-56-56L174.54 240l56 56l-56 56L200 377.46l56-56l56 56L337.46 352l-56-56l56-56z"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode(
  "path",
  {
    d: "M64 160l29.74 282.51A24 24 0 0 0 117.61 464h276.78a24 24 0 0 0 23.87-21.49L448 160zm248 217.46l-56-56l-56 56L174.54 352l56-56l-56-56L200 214.54l56 56l56-56L337.46 240l-56 56l56 56z",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_5 = /* @__PURE__ */ createBaseVNode(
  "rect",
  {
    x: "32",
    y: "48",
    width: "448",
    height: "80",
    rx: "12",
    ry: "12",
    fill: "currentColor"
  },
  null,
  -1
  /* HOISTED */
);
const _hoisted_6 = [_hoisted_2, _hoisted_3, _hoisted_4, _hoisted_5];
const TrashBinSharp = defineComponent({
  name: "TrashBinSharp",
  render: function render4(_ctx, _cache) {
    return openBlock(), createElementBlock("svg", _hoisted_1, _hoisted_6);
  }
});
var VueDrawingCanvas = /* @__PURE__ */ defineComponent({
  name: "VueDrawingCanvas",
  props: {
    strokeType: {
      type: String,
      validator: (value) => {
        return ["dash", "line", "square", "circle", "triangle", "half_triangle"].indexOf(value) !== -1;
      },
      default: () => "dash"
    },
    fillShape: {
      type: Boolean,
      default: () => false
    },
    width: {
      type: [String, Number],
      default: () => 600
    },
    height: {
      type: [String, Number],
      default: () => 400
    },
    image: {
      type: String,
      default: () => ""
    },
    eraser: {
      type: Boolean,
      default: () => false
    },
    color: {
      type: String,
      default: () => "#000000"
    },
    lineWidth: {
      type: Number,
      default: () => 5
    },
    lineCap: {
      type: String,
      validator: (value) => {
        return ["round", "square", "butt"].indexOf(value) !== -1;
      },
      default: () => "round"
    },
    lineJoin: {
      type: String,
      validator: (value) => {
        return ["miter", "round", "bevel"].indexOf(value) !== -1;
      },
      default: () => "miter"
    },
    lock: {
      type: Boolean,
      default: () => false
    },
    styles: {
      type: [Array, String, Object]
    },
    classes: {
      type: [Array, String, Object]
    },
    backgroundColor: {
      type: String,
      default: () => "#FFFFFF"
    },
    backgroundImage: {
      type: String,
      default: () => null
    },
    watermark: {
      type: Object,
      default: () => null
    },
    saveAs: {
      type: String,
      validator: (value) => {
        return ["jpeg", "png"].indexOf(value) !== -1;
      },
      default: () => "png"
    },
    canvasId: {
      type: String,
      default: () => "VueDrawingCanvas"
    },
    initialImage: {
      type: Array,
      default: () => []
    },
    additionalImages: {
      type: Array,
      default: () => []
    },
    outputWidth: {
      type: Number
    },
    outputHeight: {
      type: Number
    }
  },
  data() {
    return {
      loadedImage: null,
      drawing: false,
      context: null,
      images: [],
      strokes: {
        type: "",
        from: {
          x: 0,
          y: 0
        },
        coordinates: [],
        color: "",
        width: "",
        fill: false,
        lineCap: "",
        lineJoin: ""
      },
      guides: [],
      trash: []
    };
  },
  mounted() {
    this.setContext();
    this.$nextTick(() => {
      this.drawInitialImage();
      this.drawAdditionalImages();
    });
  },
  watch: {
    backgroundImage: function() {
      this.loadedImage = null;
    }
  },
  methods: {
    async setContext() {
      let canvas = document.querySelector("#" + this.canvasId);
      this.context = this.context ? this.context : canvas.getContext("2d");
      await this.setBackground();
    },
    drawInitialImage() {
      if (this.initialImage && this.initialImage.length > 0) {
        this.images = [].concat(this.images, this.initialImage);
        this.redraw(true);
      }
    },
    drawAdditionalImages() {
      if (this.additionalImages && this.additionalImages.length > 0) {
        let canvas = document.querySelector("#" + this.canvasId);
        this.additionalImages.forEach((watermarkObject) => {
          this.drawWatermark(canvas, this.context, watermarkObject);
        });
      }
    },
    clear() {
      this.context.clearRect(0, 0, Number(this.width), Number(this.height));
    },
    async setBackground() {
      this.clear();
      this.context.fillStyle = this.backgroundColor;
      this.context.fillRect(0, 0, Number(this.width), Number(this.height));
      await this.$nextTick(async () => {
        await this.drawBackgroundImage();
      });
      this.save();
    },
    async drawBackgroundImage() {
      if (!this.loadedImage) {
        return new Promise((resolve) => {
          if (!this.backgroundImage) {
            resolve();
            return;
          }
          const image = new Image();
          image.src = this.backgroundImage;
          image.onload = () => {
            this.context.drawImage(image, 0, 0, Number(this.width), Number(this.height));
            this.loadedImage = image;
            resolve();
          };
        });
      } else {
        this.context.drawImage(this.loadedImage, 0, 0, Number(this.width), Number(this.height));
      }
    },
    getCoordinates(event) {
      let x, y;
      if (event.touches && event.touches.length > 0) {
        let canvas = document.querySelector("#" + this.canvasId);
        let rect = canvas.getBoundingClientRect();
        x = event.touches[0].clientX - rect.left;
        y = event.touches[0].clientY - rect.top;
      } else {
        x = event.offsetX;
        y = event.offsetY;
      }
      return {
        x,
        y
      };
    },
    startDraw(event) {
      if (!this.lock) {
        this.drawing = true;
        let coordinate = this.getCoordinates(event);
        this.strokes = {
          type: this.eraser ? "eraser" : this.strokeType,
          from: coordinate,
          coordinates: [],
          color: this.eraser ? this.backgroundColor : this.color,
          width: this.lineWidth,
          fill: this.eraser || this.strokeType === "dash" || this.strokeType === "line" ? false : this.fillShape,
          lineCap: this.lineCap,
          lineJoin: this.lineJoin
        };
        this.guides = [];
      }
    },
    draw(event) {
      if (this.drawing) {
        if (!this.context) {
          this.setContext();
        }
        let coordinate = this.getCoordinates(event);
        if (this.eraser || this.strokeType === "dash") {
          this.strokes.coordinates.push(coordinate);
          this.drawShape(this.context, this.strokes, false);
        } else {
          switch (this.strokeType) {
            case "line":
              this.guides = [{
                x: coordinate.x,
                y: coordinate.y
              }];
              break;
            case "square":
              this.guides = [{
                x: coordinate.x,
                y: this.strokes.from.y
              }, {
                x: coordinate.x,
                y: coordinate.y
              }, {
                x: this.strokes.from.x,
                y: coordinate.y
              }, {
                x: this.strokes.from.x,
                y: this.strokes.from.y
              }];
              break;
            case "triangle":
              let center = Math.floor((coordinate.x - this.strokes.from.x) / 2) < 0 ? Math.floor((coordinate.x - this.strokes.from.x) / 2) * -1 : Math.floor((coordinate.x - this.strokes.from.x) / 2);
              let width = this.strokes.from.x < coordinate.x ? this.strokes.from.x + center : this.strokes.from.x - center;
              this.guides = [{
                x: coordinate.x,
                y: this.strokes.from.y
              }, {
                x: width,
                y: coordinate.y
              }, {
                x: this.strokes.from.x,
                y: this.strokes.from.y
              }];
              break;
            case "half_triangle":
              this.guides = [{
                x: coordinate.x,
                y: this.strokes.from.y
              }, {
                x: this.strokes.from.x,
                y: coordinate.y
              }, {
                x: this.strokes.from.x,
                y: this.strokes.from.y
              }];
              break;
            case "circle":
              let radiusX = this.strokes.from.x - coordinate.x < 0 ? (this.strokes.from.x - coordinate.x) * -1 : this.strokes.from.x - coordinate.x;
              this.guides = [{
                x: this.strokes.from.x > coordinate.x ? this.strokes.from.x - radiusX : this.strokes.from.x + radiusX,
                y: this.strokes.from.y
              }, {
                x: radiusX,
                y: radiusX
              }];
              break;
          }
          this.drawGuide(true);
        }
      }
    },
    drawGuide(closingPath) {
      this.redraw(true);
      this.$nextTick(() => {
        this.context.strokeStyle = this.color;
        this.context.lineWidth = 1;
        this.context.lineJoin = this.lineJoin;
        this.context.lineCap = this.lineCap;
        this.context.beginPath();
        this.context.setLineDash([15, 15]);
        if (this.strokes.type === "circle") {
          this.context.ellipse(this.guides[0].x, this.guides[0].y, this.guides[1].x, this.guides[1].y, 0, 0, Math.PI * 2);
        } else {
          this.context.moveTo(this.strokes.from.x, this.strokes.from.y);
          this.guides.forEach((coordinate) => {
            this.context.lineTo(coordinate.x, coordinate.y);
          });
          if (closingPath) {
            this.context.closePath();
          }
        }
        this.context.stroke();
      });
    },
    drawShape(context, strokes, closingPath) {
      context.strokeStyle = strokes.color;
      context.fillStyle = strokes.color;
      context.lineWidth = strokes.width;
      context.lineJoin = strokes.lineJoin === void 0 ? this.lineJoin : strokes.lineJoin;
      context.lineCap = strokes.lineCap === void 0 ? this.lineCap : strokes.lineCap;
      context.beginPath();
      context.setLineDash([]);
      if (strokes.type === "circle") {
        context.ellipse(strokes.coordinates[0].x, strokes.coordinates[0].y, strokes.coordinates[1].x, strokes.coordinates[1].y, 0, 0, Math.PI * 2);
      } else {
        context.moveTo(strokes.from.x, strokes.from.y);
        strokes.coordinates.forEach((stroke) => {
          context.lineTo(stroke.x, stroke.y);
        });
        if (closingPath) {
          context.closePath();
        }
      }
      if (strokes.fill) {
        context.fill();
      } else {
        context.stroke();
      }
    },
    stopDraw() {
      if (this.drawing) {
        this.strokes.coordinates = this.guides.length > 0 ? this.guides : this.strokes.coordinates;
        this.images.push(this.strokes);
        this.redraw(true);
        this.drawing = false;
        this.trash = [];
      }
    },
    reset() {
      if (!this.lock) {
        this.images = [];
        this.strokes = {
          type: "",
          coordinates: [],
          color: "",
          width: "",
          fill: false,
          lineCap: "",
          lineJoin: ""
        };
        this.guides = [];
        this.trash = [];
        this.redraw(true);
      }
    },
    undo() {
      if (!this.lock) {
        let strokes = this.images.pop();
        if (strokes) {
          this.trash.push(strokes);
          this.redraw(true);
        }
      }
    },
    redo() {
      if (!this.lock) {
        let strokes = this.trash.pop();
        if (strokes) {
          this.images.push(strokes);
          this.redraw(true);
        }
      }
    },
    async redraw(output) {
      output = typeof output !== "undefined" ? output : true;
      await this.setBackground().then(() => {
        this.drawAdditionalImages();
      }).then(() => {
        let baseCanvas = document.createElement("canvas");
        let baseCanvasContext = baseCanvas.getContext("2d");
        baseCanvas.width = Number(this.width);
        baseCanvas.height = Number(this.height);
        if (baseCanvasContext) {
          this.images.forEach((stroke) => {
            if (baseCanvasContext) {
              baseCanvasContext.globalCompositeOperation = stroke.type === "eraser" ? "destination-out" : "source-over";
              if (stroke.type !== "circle" || stroke.type === "circle" && stroke.coordinates.length > 0) {
                this.drawShape(baseCanvasContext, stroke, stroke.type === "eraser" || stroke.type === "dash" || stroke.type === "line" ? false : true);
              }
            }
          });
          this.context.drawImage(baseCanvas, 0, 0, Number(this.width), Number(this.height));
        }
      }).then(() => {
        if (output) {
          this.save();
        }
      });
    },
    wrapText(context, text, x, y, maxWidth, lineHeight) {
      const newLineRegex = /(\r\n|\n\r|\n|\r)+/g;
      const whitespaceRegex = /\s+/g;
      var lines = text.split(newLineRegex).filter((word) => word.length > 0);
      for (let lineNumber = 0; lineNumber < lines.length; lineNumber++) {
        var words = lines[lineNumber].split(whitespaceRegex).filter((word) => word.length > 0);
        var line = "";
        for (var n = 0; n < words.length; n++) {
          var testLine = line + words[n] + " ";
          var metrics = context.measureText(testLine);
          var testWidth = metrics.width;
          if (testWidth > maxWidth && n > 0) {
            if (this.watermark && this.watermark.fontStyle && this.watermark.fontStyle.drawType && this.watermark.fontStyle.drawType === "stroke") {
              context.strokeText(line, x, y);
            } else {
              context.fillText(line, x, y);
            }
            line = words[n] + " ";
            y += lineHeight;
          } else {
            line = testLine;
          }
        }
        if (this.watermark && this.watermark.fontStyle && this.watermark.fontStyle.drawType && this.watermark.fontStyle.drawType === "stroke") {
          context.strokeText(line, x, y);
        } else {
          context.fillText(line, x, y);
        }
        y += words.length > 0 ? lineHeight : 0;
      }
    },
    save() {
      let canvas = document.querySelector("#" + this.canvasId);
      if (this.watermark) {
        let temp = document.createElement("canvas");
        let ctx = temp.getContext("2d");
        if (ctx) {
          temp.width = Number(this.width);
          temp.height = Number(this.height);
          ctx.drawImage(canvas, 0, 0, Number(this.width), Number(this.height));
          this.drawWatermark(temp, ctx, this.watermark);
        }
      } else {
        let temp = document.createElement("canvas");
        let tempCtx = temp.getContext("2d");
        let tempWidth = this.outputWidth === void 0 ? this.width : this.outputWidth;
        let tempHeight = this.outputHeight === void 0 ? this.height : this.outputHeight;
        temp.width = Number(tempWidth);
        temp.height = Number(tempHeight);
        if (tempCtx) {
          tempCtx.drawImage(canvas, 0, 0, Number(tempWidth), Number(tempHeight));
          this.$emit("update:image", temp.toDataURL("image/" + this.saveAs, 1));
          return temp.toDataURL("image/" + this.saveAs, 1);
        }
      }
    },
    drawWatermark(canvas, ctx, watermark) {
      if (watermark.type === "Image") {
        let imageWidth = watermark.imageStyle ? watermark.imageStyle.width ? watermark.imageStyle.width : Number(this.width) : Number(this.width);
        let imageHeight = watermark.imageStyle ? watermark.imageStyle.height ? watermark.imageStyle.height : Number(this.height) : Number(this.height);
        const image = new Image();
        image.src = watermark.source;
        image.onload = () => {
          if (watermark && ctx) {
            ctx.drawImage(image, watermark.x, watermark.y, Number(imageWidth), Number(imageHeight));
          }
          let temp = document.createElement("canvas");
          let tempCtx = temp.getContext("2d");
          let tempWidth = this.outputWidth === void 0 ? this.width : this.outputWidth;
          let tempHeight = this.outputHeight === void 0 ? this.height : this.outputHeight;
          temp.width = Number(tempWidth);
          temp.height = Number(tempHeight);
          if (tempCtx) {
            tempCtx.drawImage(canvas, 0, 0, Number(tempWidth), Number(tempHeight));
            this.$emit("update:image", temp.toDataURL("image/" + this.saveAs, 1));
            return temp.toDataURL("image/" + this.saveAs, 1);
          }
        };
      } else if (watermark.type === "Text") {
        let font = watermark.fontStyle ? watermark.fontStyle.font ? watermark.fontStyle.font : "20px serif" : "20px serif";
        let align = watermark.fontStyle ? watermark.fontStyle.textAlign ? watermark.fontStyle.textAlign : "start" : "start";
        let baseline = watermark.fontStyle ? watermark.fontStyle.textBaseline ? watermark.fontStyle.textBaseline : "alphabetic" : "alphabetic";
        let color = watermark.fontStyle ? watermark.fontStyle.color ? watermark.fontStyle.color : "#000000" : "#000000";
        ctx.font = font;
        ctx.textAlign = align;
        ctx.textBaseline = baseline;
        if (watermark.fontStyle && watermark.fontStyle.rotate) {
          let centerX, centerY;
          if (watermark.fontStyle && watermark.fontStyle.width) {
            centerX = watermark.x + Math.floor(watermark.fontStyle.width / 2);
          } else {
            centerX = watermark.x;
          }
          if (watermark.fontStyle && watermark.fontStyle.lineHeight) {
            centerY = watermark.y + Math.floor(watermark.fontStyle.lineHeight / 2);
          } else {
            centerY = watermark.y;
          }
          ctx.translate(centerX, centerY);
          ctx.rotate(watermark.fontStyle.rotate * Math.PI / 180);
          ctx.translate(centerX * -1, centerY * -1);
        }
        if (watermark.fontStyle && watermark.fontStyle.drawType && watermark.fontStyle.drawType === "stroke") {
          ctx.strokeStyle = watermark.fontStyle.color;
          if (watermark.fontStyle && watermark.fontStyle.width) {
            this.wrapText(ctx, watermark.source, watermark.x, watermark.y, watermark.fontStyle.width, watermark.fontStyle.lineHeight);
          } else {
            ctx.strokeText(watermark.source, watermark.x, watermark.y);
          }
        } else {
          ctx.fillStyle = color;
          if (watermark.fontStyle && watermark.fontStyle.width) {
            this.wrapText(ctx, watermark.source, watermark.x, watermark.y, watermark.fontStyle.width, watermark.fontStyle.lineHeight);
          } else {
            ctx.fillText(watermark.source, watermark.x, watermark.y);
          }
        }
        let temp = document.createElement("canvas");
        let tempCtx = temp.getContext("2d");
        let tempWidth = this.outputWidth === void 0 ? this.width : this.outputWidth;
        let tempHeight = this.outputHeight === void 0 ? this.height : this.outputHeight;
        temp.width = Number(tempWidth);
        temp.height = Number(tempHeight);
        if (tempCtx) {
          tempCtx.drawImage(canvas, 0, 0, Number(tempWidth), Number(tempHeight));
          this.$emit("update:image", temp.toDataURL("image/" + this.saveAs, 1));
          return temp.toDataURL("image/" + this.saveAs, 1);
        }
      }
    },
    isEmpty() {
      return this.images.length > 0 ? false : true;
    },
    getAllStrokes() {
      return this.images;
    }
  },
  render() {
    return h("canvas", {
      id: this.canvasId,
      height: Number(this.height),
      width: Number(this.width),
      style: {
        "touchAction": "none",
        // @ts-ignore
        ...this.styles
      },
      class: this.classes,
      onMousedown: ($event) => this.startDraw($event),
      onMousemove: ($event) => this.draw($event),
      onMouseup: () => this.stopDraw(),
      onMouseleave: () => this.stopDraw(),
      onTouchstart: ($event) => this.startDraw($event),
      onTouchmove: ($event) => this.draw($event),
      onTouchend: () => this.stopDraw(),
      onTouchleave: () => this.stopDraw(),
      onTouchcancel: () => this.stopDraw(),
      onPointerdown: ($event) => this.startDraw($event),
      onPointermove: ($event) => this.draw($event),
      onPointerup: () => this.stopDraw(),
      onPointerleave: () => this.stopDraw(),
      onPointercancel: () => this.stopDraw()
    });
  }
});
export {
  ArrowUndoSharp as A,
  BrushSharp as B,
  TrashBinSharp as T,
  VueDrawingCanvas as V,
  ArrowRedoSharp as a
};
