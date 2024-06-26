
class LayerFIFOQueue {
    constructor(layer, capacity = 64, KeysBackingArrayType = Uint32Array) {
        this.capacity = capacity;
        this.layer = layer;
        this.childIndices = new Set();
        this.length = 0;
    }

    add(child) {
        if (this.childIndices.has(child.ann.id)) {
            return;
        }

        if (this.isFull()) {
            let pop_child = this.layer.children[0];  // O(n) not O(1)
            this.remove(pop_child);
        }
//         let area = (child.ann['x1'] - child.ann['x0']) * (child.ann['y1'] - child.ann['y0']);
        this.layer.add(child);
        this.childIndices.add(child.ann.id);
        this.length++;
    }

    remove(child) {
        this.childIndices.delete(child.ann.id);
        child.destroy();
        this.length--;
    }
    
    children() {
        return this.layer.children;
    }

    destroyChildren() {
        this.layer.destroyChildren();
        this.length = 0;
        this.childIndices.clear();
    }

    isEmpty() {
        return this.length === 0;
    }

    isFull() {
        return this.length === this.capacity;
    }
}


class IViewerAnnotation {
    static count = 0;
    constructor(viewer, configs) {
//         if (!window.OpenSeadragon) {
//             console.error('[openseadragon-konva-overlay] requires OpenSeadragon');
//             return;
//         }
        this._viewer = viewer;
        this.cfs = configs || {};
        this._id = configs?.id || 'osd-overlaycanvas-' + (++IViewerAnnotation.count);
//         this.cfs.enablePointerEvents = window.PointerEvent != null;
        if (this.cfs?.disableClickToZoom) {
            this._viewer.gestureSettingsMouse.clickToZoom = false;
        }

        // Build canvas div and Konva Stage/Layers
        this._canvasdiv = document.createElement('div');
        this._canvasdiv.setAttribute('id', this._id);
        this._canvasdiv.style.position = 'absolute';
        this._canvasdiv.style.left = 0;
        this._canvasdiv.style.top = 0;
        this._canvasdiv.style.width = '100%';
        this._canvasdiv.style.height = '100%';
        this._viewer.canvas.appendChild(this._canvasdiv);

        this._containerWidth = 0;
        this._containerHeight = 0;
        this.resize();

        // Create a Konva stage in canvas div
        this._konvaStage = new Konva.Stage({
            container: this._id,
//             width: this._viewer.container.clientWidth,
//             height: this._viewer.container.clientHeight,
//             draggable: false,  // Disable draggable because default click is tracked by OSD
//             opacity: 1.0,
        });
        this.registerKonvaActions();

        // Add Konva Layers
        const layerCfgs = this.cfs?.layers || [{'id': 'konvaLayer-0', 'capacity': 512}];
        this.layerQueues = {};
        layerCfgs.forEach((cfg, index) => {
            let id = cfg?.id || `konvaLayer-${index}`;
            let capacity = cfg?.capacity || null;
            this.addLayer(id=id, capacity=capacity);
        });
        
        // Add a colorPalette to Konva if given.
        let colorPalette = this.cfs?.colorPalette || {};
        this.colorPalette = new ColorPalette(colorPalette);
        
        // Add Annotorious Layer
        let widgets = this.cfs?.widgets || [];
        this._annotoriousLayer = OpenSeadragon.Annotorious(viewer, {
            locale: 'auto',
            allowEmpty: true,
            widgets: widgets,
        });

        // Add Annotorious SelectorPack and ToolBars
        let toolCfgs = this.cfs?.drawingTools || {'tools': ['rect', 'polygon', 'circle', 'ellipse', 'freehand']};
        if (toolCfgs) {
            try {
                Annotorious.BetterPolygon(this._annotoriousLayer);
            } catch (error) {  // Package is not available
                console.log("Annotorious.BetterPolygon is not available. Use default polygon.");
            }
            Annotorious.SelectorPack(this._annotoriousLayer, {tools: toolCfgs.tools});
            if (toolCfgs?.container) {
                let barCfgs = {'drawingTools': toolCfgs.tools || toolCfgs?.drawingTools};
                if (toolCfgs?.withMouse) barCfgs['withMouse'] = true;
                if (toolCfgs?.withLabel) barCfgs['withLabel'] = true;
                if (toolCfgs?.withTooltip) barCfgs['withTooltip'] = true;
                if (toolCfgs?.infoElement) barCfgs['infoElement'] = toolCfgs.infoElement;
                console.log(barCfgs);
                Annotorious.Toolbar(this._annotoriousLayer, toolCfgs.container, barCfgs);
            }
        }

        // Add A pool of modifying objects. TODO: use array in future
        this._modifingNode = null;

        this.activeAnnotators = new Set();
        
        // Bind all event functions
        this.render = this.render.bind(this);
        this.renderActive = this.renderActive.bind(this);

        this.recordViewport = this.recordViewport.bind(this);
        this.recordInterval = this.recordInterval.bind(this);
    }

    registerKonvaActions() {
        // Update viewport
        this._viewer.addHandler('update-viewport', () => {
            this.resize();
            this.resizeCanvas();
        });

        // Resize the konva overlay when the viewer or window changes size
        this._viewer.addHandler('open', () => {
            this.resize();
            this.resizeCanvas();
        });

        window.addEventListener('resize', () => {
            this.resize();
            this.resizeCanvas();
        });

        // window.addEventListener('scroll', function (event) {
        //     // this.resize();
        //     // this.resizeCanvas();
        //     console.log("scoll down/up")
        //     event.preventDefault();
        // }, { passive: true });
    }

    buildConnections(createDB, getAnnotators, getLabels, insert, read, update, deleteanno, search, stream, count) {
        this.APIs = {
            annoCreateDBAPI: createDB,
            annoGetAnnotators: getAnnotators,
            annoGetLabels: getLabels,
            annoInsertAPI: insert,
            annoReadAPI: read,
            annoUpdateAPI: update,
            annoDeleteAPI: deleteanno,
            annoSearchAPI: search,
            annoStreamAPI: stream,
            annoCountAnnos: count,
        }
        
        //after loading the slide, call api to create its db
        fetch(this.APIs.annoCreateDBAPI, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
        }).catch(error => {
            console.error('There was a when creating database:', error);
        });

        this.webSocket = new WebSocket(this.APIs.annoStreamAPI);
//         this._viewer.addHandler('viewport-change', event => {
//             this.render();
//         });
    }

    streamAnnotations(layerQueue, query) {
        // skip message if null, missing(undefined) is allowed
        let tag = true;
        if ('annotator' in query) {
            if (query['annotator'] === null) tag = false;
            if (Array.isArray(query['annotator']) && query['annotator'].length === 0) tag = false;
        }
        if ('label' in query) {
            if (query['label'] === null) tag = false;
            if (Array.isArray(query['label']) && query['label'].length === 0) tag = false;
        }

        if (tag) {
            // this.webSocket.send(JSON.stringify(query));
            sendMessageWithRetry(this.webSocket, query);
            this.webSocket.onmessage = (event) => {
                let ann = JSON.parse(event.data);
                // console.log("return annotation", ann)
                let item = this.createKonvaItem(ann);
                layerQueue.add(item);
            }
//             layerQueue.layer.batchDraw();
        }
    };

    //= streamannotation function
    render(cfgs={}) {
        let bbox = getCurrentViewport(this._viewer);
        let query;
        if (bbox) {
            let windowArea = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            query = {
                'x0': bbox.x0,
                'y0': bbox.y0,
                'x1': bbox.x1,
                'y1': bbox.y1,
                'min_box_area': windowArea * 0.0001,
                'max_box_area': windowArea,
                ...cfgs,
            }
        } else {  // if no bbox information in query
            query = {...cfgs};
        }
        // console.log('Search query:', query);
        this.streamAnnotations(this.getLayerQueue(), query);
        // this._konvaStage.draw();
    }

    renderActive() {
        this.render({'annotator': Array.from(this.activeAnnotators)});
    } 

    draw() {
        this._viewer.addHandler('viewport-change', this.renderActive);
    }

    hide() {
        this._viewer.removeHandler('viewport-change', this.renderActive);
    }

    updateAnnotators(annotators) {
        let newAnnotators = new Set(annotators);
//         console.log("Incoming Annotators: ", newAnnotators);

        // Remove annotators
        // const deletedIds = this.activeAnnotators.difference(newAnnotators);
        const deletedIds = new Set([...this.activeAnnotators].filter(x => !newAnnotators.has(x)));
        console.log("deletedids", deletedIds)
        if (deletedIds.size > 0) {
//             console.log("Delete Annotators: ", deletedIds);
            let layerQueue = this.getLayerQueue();
            let popNodes = layerQueue.children().filter(
                child => deletedIds.has(child.ann.annotator)
            );
            popNodes.forEach(node => {
                layerQueue.remove(node);
            });
        }

        // Add annotators
        // const addIds s= newAnnotators.difference(this.activeAnnotators);
        const addIds = new Set([...newAnnotators].filter(x => !this.activeAnnotators.has(x)));
        console.log("addIds", addIds)
        if (addIds.size > 0) {
//             console.log("Add Annotators: ", addIds);
            this.render({'annotator': Array.from(addIds)});
        }

        this.activeAnnotators = newAnnotators;
//         console.log("Updated Annotators: ", this.activeAnnotators);
    }

    // Add a new annotator into activeAnnotators
    addAnnotator(annotator) {
        if (!this.activeAnnotators.has(annotator)) {
            this.activeAnnotators.add(annotator);
            this.render({'annotator': annotator});
        }
    }

    // Remove a annotator from activeAnnotators
    removeAnnotator(annotator) {
        if (this.activeAnnotators.has(annotator)) {
            this.activeAnnotators.delete(annotator);
            let layerQueue = this.getLayerQueue();
            let popNodes = layerQueue.children().filter(
                child => child.ann.annotator === annotator
            );
            popNodes.forEach(node => {
                layerQueue.remove(node);
            });
        }
    }

    enableEditing(userId) {
        this.userId = userId;
        
        // Enter modify mode: hide object and create an annotorious object
        this._viewer.addHandler("canvas-click", event => {
            let selected = this._annotoriousLayer.getSelected();
            if (!selected) {  // make sure no existing selection
                // let item = this._konvaStage.getIntersection({ x: event.position.x, y: event.position.y }); // more efficient but can't find small item which is under large item
                let items = this._konvaStage.getAllIntersections({ x: event.position.x, y: event.position.y });
                let smallestValue = Infinity, item=items[0];
                for (let i of items) {
                    // Calculate the absolute differences
                    let xDiff = Math.abs(i.ann.x1 - i.ann.x0);
                    let yDiff = Math.abs(i.ann.y1 - i.ann.y0);
                
                    // Calculate the product of absolute differences
                    let product = xDiff * yDiff;
                
                    // Check if the current product is smaller than the smallest value found so far
                    if (product < smallestValue) {
                        smallestValue = product;
                        item = i;
                    }
                }

                var {x0, y0, x1, y1} = getCurrentViewport(this._viewer);
                if (item && item.ann.x0 >= x0 && item.ann.y0 >= y0 && item.ann.x1 <= x1 && item.ann.y1 <= y1) {
                    console.log('Mouse click on Konva shape and hide:', item.ann.id, item);
                    this._viewer.gestureSettingsMouse.clickToZoom = false;
                    this._modifingNode?.show();
                    this._modifingNode = item;
                    item.hide();

                    let svgAnno = konva2w3c(item)[0];
                    console.log('After convert Konva shape to W3C svg:', svgAnno);
                    this._annotoriousLayer.addAnnotation(svgAnno);
                    this._annotoriousLayer.selectAnnotation(svgAnno);
                } else {
                    this._viewer.gestureSettingsMouse.clickToZoom = true;
                }
            }
        });
        
        this._annotoriousLayer.on('cancelSelected', annotation => {
            this._modifingNode?.show();
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });

        // Update konva object with annotorious changes
        this._annotoriousLayer.on('createAnnotation', annotation => {
//             let query = w3c2konva(annotation, this.userId);  // JSON.stringify(annotation)
            let query = w3c2konva(annotation);  // JSON.stringify(annotation)
            query['annotator'] = this.userId.toString();
//             console.log("Enter createAnnotation", query);
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.cancelSelected();
            this._annotoriousLayer.clearAnnotations();

            let api = this.APIs.annoInsertAPI;
            console.log('annotation query: ', query, api)
            createAnnotation(api, query).then(ann => {
                let item = this.createKonvaItem(ann);
                this.getLayerQueue().add(item);
                this.addAnnotator(this.userId) //display the login user's annotation after creating a new annotation
                //this.activeAnnotators need to filter out model annotator
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators}, this.colorPalette);
            });
        });

        this._annotoriousLayer.on('updateAnnotation', (annotation, previous) => {
            let item = this._modifingNode;
//             let query = w3c2konva(annotation, this.userId);
            let query = w3c2konva(annotation);
            query['annotator'] = this.userId.toString();
//             console.log("Enter updateAnnotation", item.ann.id, item, query);

            let api = this.APIs.annoUpdateAPI + item.ann.id;
            updateAnnotation(api, query).then(ann => {
                item = this.updateKonvaItem(ann, item);
                item.show();
                this.addAnnotator(this.userId)
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                  console.log()
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators}, this.colorPalette);
            });
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });

        this._annotoriousLayer.on('deleteAnnotation', annotation => {
            let item = this._modifingNode;
            let api = this.APIs.annoDeleteAPI + item.ann.id;
            deleteAnnotation(api).then(resp => {
                console.log(resp);
                this.getLayerQueue().remove(item);
                const filteredActiveAnnotators = [];
                this.activeAnnotators.forEach(item => {
                    if (!isNaN(item)) {
                        filteredActiveAnnotators.push(item);
                    }
                  });
                drawNUpdateDatatable(this.APIs.annoSearchAPI, {"annotator": filteredActiveAnnotators}, this.colorPalette);
            });
            this._annotoriousLayer.cancelSelected();
            this._annotoriousLayer.removeAnnotation(annotation);
            this._annotoriousLayer.clearAnnotations();
            this._modifingNode = null;
        });
    }

    addLayer(id, capacity=null) {
        let layer = new Konva.Layer();
        this._konvaStage.add(layer);
        this.layerQueues[id] = new LayerFIFOQueue(layer, capacity=capacity);
        console.log("Create new layer: ", capacity, this.layerQueues[id]);
    }

    konvaStage() {
        return this._konvaStage;
    }

    konvaLayers() {
        return this._konvaStage.getLayers();
    }
    
    getLayerQueue(id='konvaLayer-0') {
        return this.layerQueues[id];
    }

    clear() {
        for (let key in this.layerQueues) {
            this.layerQueues[key].destroyChildren();
        }
        this._konvaStage.clearAll();
    }

    resize() {
        if (this._containerWidth !== this._viewer.container.clientWidth) {
            this._containerWidth = this._viewer.container.clientWidth;
            this._canvasdiv.setAttribute('width', this._containerWidth);
        }

        if (this._containerHeight !== this._viewer.container.clientHeight) {
            this._containerHeight = this._viewer.container.clientHeight;
            this._canvasdiv.setAttribute('height', this._containerHeight);
        }
    }
    
    resizeCanvas() {
        let origin = new OpenSeadragon.Point(0, 0);
        let viewportZoom = this._viewer.viewport.getZoom(true);
        this._konvaStage.setWidth(this._containerWidth);
        this._konvaStage.setHeight(this._containerHeight);

        let imageSize = this._viewer.world.getItemAt(0).getContentSize();
        let zoom = this._viewer.viewport._containerInnerSize.x * viewportZoom / imageSize.x;
        this._konvaStage.scale({ x: zoom, y: zoom });

        let viewportWindowPoint = this._viewer.viewport.viewportToWindowCoordinates(origin);
        let x = Math.round(viewportWindowPoint.x);
        let y = Math.round(viewportWindowPoint.y);
        let canvasOffset = this._canvasdiv.getBoundingClientRect();

        let pageScroll = OpenSeadragon.getPageScroll();
        this._konvaStage.x(x - canvasOffset.x - window.scrollX);
        this._konvaStage.y(y - canvasOffset.y - window.scrollY);
        this._konvaStage.draw();
    }

    createKonvaItem(ann) {
        let itemCfgs = parseDatabaseAnnotation(ann, this.colorPalette);
        let item;
        if (itemCfgs["shape"] === "polygon") {
            item = new Konva.Line(itemCfgs);
        } else if (itemCfgs["shape"] == "rect") {
            item = new Konva.Rect(itemCfgs);
        } else {
            item = new Konva.Ellipse(itemCfgs);
        }
        item.ann = ann;

        return item;
    }

    updateKonvaItem(ann, item) {
        let itemCfgs = parseDatabaseAnnotation(ann, this.colorPalette);
        item.setAttrs(itemCfgs);
        item.ann = ann;

        return item;
    }

    removeAllShapes() {
        let layerQueue = this.getLayerQueue();
        layerQueue.destroyChildren();
    }

    recordViewport() {
        // static/utils.js getCurrentViewport
        // api: your route in FastAPI
        let api = this.APIs.annoInsertAPI;
        // other_params = {'annotator', 'timestamp', etc.}
        
        let bbox = getCurrentViewport(this._viewer);
        let query = {
            'x0': bbox.x0,
            'y0': bbox.y0,
            'x1': bbox.x1,
            'y1': bbox.y1,
            //...other_params,
        }
        console.log("Recording Viewport", bbox, query)
        // static/utils.js createAnnotation
        //createAnnotation(api, query);
        return query;
    }
    
    startRecord() {
        // use lamda function to parse parameters
        this._viewer.addHandler('viewport-change', this.recordInterval);
        //this.recordViewport()
        let temp = this.recordViewport()
        let timestamp = Date.now()
        console.log(temp, timestamp)
        temp.timestamp = timestamp
        recorderQueue.push(temp)
        // if(intervalId == null){
        //     intervalId = setInterval(this.recordInterval, 1000)
        //     console.log("Interval STARTED")
        // }
        console.log("Started Recording")
        document.getElementById('recorder').innerHTML = 'Stop Recording'
    }
    async stopRecord() {
        this._viewer.removeHandler('viewport-change', this.recordInterval);
        // if (intervalId !== null) {
        //     clearInterval(intervalId);
        //     intervalId = null;
        //     console.log("Interval stopped");
        // }
        document.getElementById('recorder').innerHTML = 'Record'
        console.log("Stopped Recording", recorderQueue)
        const response = await fetch("http://localhost:9050/upload", {'method': 'GET'})
        const data = await response.json()
        // .then(response => response.json())
        // .then(data => {
            // Use the variable in JavaScript
        const path = data.file_path;
        const text = data.text;
        const timestamp = data.timestamp
        this.recordAnnotations(text)
            // let index = 0;
            // console.log(recorderQueue)
            // for(let i = 0;i<recorderQueue.length-1; i++){
            //     if(text[index]['timestamp'][0]>=recorderQueue[i]['timestamp']&&text[index]['timestamp'][0]<=recorderQueue[i+1]['timestamp']){
            //         recorderQueue[i].annotation = text[index]['text']
            //         annotationList.push(recorderQueue[i])
            //         index++;
            //     }
            // }
            // // Update the HTML element with the variable
            // console.log(path, text)
            // console.log(recorderQueue)
        // })
        // .catch(error => console.error('Error fetching the variable:', error));
        console.log("async")
        recorderQueue = []
        annotationList = []
    }
    recordAnnotations(text) {
        let index = 0;
        console.log(recorderQueue)
        let i = 0
        recorderQueue.push({'timestamp': 99999999999999})
        while(i<recorderQueue.length-1){
            //console.log(recorderQueue[i])
            if(text[index]['timestamp'][0]>=recorderQueue[i]['timestamp']&&text[index]['timestamp'][0]<=recorderQueue[i+1]['timestamp']){
                if('description' in recorderQueue[i]){
                    recorderQueue[i].description += text[index]['text']
                    console.log(recorderQueue[i], text[index]['text'])
                    annotationList.pop()
                    annotationList.push(recorderQueue[i])
                }
                else{
                    recorderQueue[i].description = text[index]['text']
                    annotationList.push(recorderQueue[i])
                }
                
                index++;
                if(index>=text.length){break;}
                continue;
            }
            i++
        }
        console.log(annotationList)
        for(let i = 0;i<annotationList.length;i++){
            let api = this.APIs.annoInsertAPI;
            createAnnotation(api, annotationList[i]);
        }
        
    }
    recordInterval() {
        let newQuery= this.recordViewport()
        let timestamp = Date.now()
        console.log(recorderQueue, recorderQueue[recorderQueue.length - 1], newQuery)
        if(JSON.stringify(recorderQueue[recorderQueue.length - 1])!==JSON.stringify(newQuery)){
            newQuery.timestamp = timestamp
            recorderQueue.push(newQuery)
            console.log("VIEWPORT CAUGHT!", newQuery)
        }
        else{
            console.log("SAME VIEWPORT")
        }
    }
}

let recorderQueue= []
let annotationList = []
// export default IViewerAnnotation;