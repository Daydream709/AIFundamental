let ROWS = 25;
let COLS = 45;

let grid = [];
let startNode = { row: 12, col: 10 };
let endNode = { row: 12, col: 35 };

let isDrawing = false;
let isMovingStart = false;
let isMovingEnd = false;
let isAnimating = false;

const gridElement = document.getElementById("grid");

// Update grid size based on user input
function updateGridSize() {
  const rowsInput = document.getElementById("grid-rows");
  const colsInput = document.getElementById("grid-cols");

  let newRows = parseInt(rowsInput.value);
  let newCols = parseInt(colsInput.value);

  // Validate and clamp values to 1-100 range
  newRows = Math.max(1, Math.min(100, newRows));
  newCols = Math.max(1, Math.min(100, newCols));

  // Update input fields with clamped values
  rowsInput.value = newRows;
  colsInput.value = newCols;

  ROWS = newRows;
  COLS = newCols;

  // Recalculate start and end positions proportionally
  startNode = {
    row: Math.floor(ROWS / 2),
    col: Math.floor(COLS / 4),
  };
  endNode = {
    row: Math.floor(ROWS / 2),
    col: Math.floor((3 * COLS) / 4),
  };

  // Clear animation state
  isAnimating = false;

  // Reinitialize grid
  initializeGrid();
}

function initializeGrid() {
  gridElement.innerHTML = "";
  grid = [];
  for (let r = 0; r < ROWS; r++) {
    let rowArray = [];
    let rowElement = document.createElement("div");
    rowElement.className = "row";
    for (let c = 0; c < COLS; c++) {
      let nodeElement = document.createElement("div");
      nodeElement.id = `node-${r}-${c}`;
      nodeElement.className = "node";
      if (r === startNode.row && c === startNode.col) nodeElement.classList.add("node-start");
      if (r === endNode.row && c === endNode.col) nodeElement.classList.add("node-end");

      nodeElement.addEventListener("mousedown", () => handleMouseDown(r, c));
      nodeElement.addEventListener("mouseenter", () => handleMouseEnter(r, c));
      nodeElement.addEventListener("mouseup", handleMouseUp);

      rowElement.appendChild(nodeElement);
      rowArray.push({ row: r, col: c, isWall: false, isVisited: false, isPath: false });
    }
    gridElement.appendChild(rowElement);
    grid.push(rowArray);
  }
}

function handleMouseDown(r, c) {
  if (isAnimating) return;
  if (r === startNode.row && c === startNode.col) {
    isMovingStart = true;
  } else if (r === endNode.row && c === endNode.col) {
    isMovingEnd = true;
  } else {
    isDrawing = true;
    toggleWall(r, c);
  }
}

function handleMouseEnter(r, c) {
  if (!isDrawing && !isMovingStart && !isMovingEnd) return;
  if (isMovingStart) {
    document.getElementById(`node-${startNode.row}-${startNode.col}`).classList.remove("node-start");
    startNode = { row: r, col: c };
    document.getElementById(`node-${r}-${c}`).classList.add("node-start");
  } else if (isMovingEnd) {
    document.getElementById(`node-${endNode.row}-${endNode.col}`).classList.remove("node-end");
    endNode = { row: r, col: c };
    document.getElementById(`node-${r}-${c}`).classList.add("node-end");
  } else if (isDrawing) {
    toggleWall(r, c);
  }
}

function handleMouseUp() {
  isDrawing = false;
  isMovingStart = false;
  isMovingEnd = false;
}

function toggleWall(r, c) {
  if ((r === startNode.row && c === startNode.col) || (r === endNode.row && c === endNode.col)) return;
  grid[r][c].isWall = !grid[r][c].isWall;
  const nodeEl = document.getElementById(`node-${r}-${c}`);
  if (grid[r][c].isWall) nodeEl.classList.add("node-wall");
  else nodeEl.classList.remove("node-wall");
}

function clearPath() {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      grid[r][c].isVisited = false;
      grid[r][c].isPath = false;
      grid[r][c].previousNode = null;
      grid[r][c].backwardPreviousNode = null;
      grid[r][c].visitedFromStart = false;
      grid[r][c].visitedFromEnd = false;
      const nodeEl = document.getElementById(`node-${r}-${c}`);
      nodeEl.classList.remove("node-visited", "node-path");
    }
  }
}

function clearBoard() {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      grid[r][c].isWall = false;
      const nodeEl = document.getElementById(`node-${r}-${c}`);
      nodeEl.classList.remove("node-wall", "node-visited", "node-path");
    }
  }
}

function generateRandomMaze() {
  clearBoard();
  clearPath();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (Math.random() < 0.3) {
        if ((r !== startNode.row || c !== startNode.col) && (r !== endNode.row || c !== endNode.col)) {
          grid[r][c].isWall = true;
          document.getElementById(`node-${r}-${c}`).classList.add("node-wall");
        }
      }
    }
  }
}

function getNeighbors(node) {
  const neighbors = [];
  const { row, col } = node;
  if (row > 0) neighbors.push(grid[row - 1][col]);
  if (row < ROWS - 1) neighbors.push(grid[row + 1][col]);
  if (col > 0) neighbors.push(grid[row][col - 1]);
  if (col < COLS - 1) neighbors.push(grid[row][col + 1]);
  return neighbors.filter((n) => !n.isWall);
}

// ---------------- ALGORITHMS ---------------- //
function runAlgorithm(algo) {
  clearPath();
  const start = grid[startNode.row][startNode.col];
  const end = grid[endNode.row][endNode.col];
  let visitedNodesInOrder = [];
  let path = [];
  const startTime = performance.now();

  if (algo === "bfs") bfs(start, end, visitedNodesInOrder);
  else if (algo === "dfs") dfs(start, end, visitedNodesInOrder);
  else if (algo === "dfs_brute") dfsBruteForce(start, end, visitedNodesInOrder);
  else if (algo === "bidirectional_bfs") bidirectionalBFS(start, end, visitedNodesInOrder);
  else if (algo === "dijkstra") dijkstra(start, end, visitedNodesInOrder);
  else if (algo === "astar") astar(start, end, visitedNodesInOrder, 1);
  else if (algo === "weighted_astar")
    astar(start, end, visitedNodesInOrder, 2.5); // Weight for heuristic
  else if (algo === "greedy") greedyBestFirstSearch(start, end, visitedNodesInOrder);

  const endTime = performance.now();
  path = getShortestPath(end);

  document.getElementById("stat-time").innerText = (endTime - startTime).toFixed(2);
  document.getElementById("stat-visited").innerText = visitedNodesInOrder.length;
  document.getElementById("stat-path").innerText = path.length > 1 ? path.length - 1 : 0;

  animateAlgorithm(visitedNodesInOrder, path);
}

function bfs(start, end, visitedNodesInOrder) {
  let queue = [start];
  start.isVisited = true;
  while (queue.length) {
    let curr = queue.shift();
    visitedNodesInOrder.push(curr);
    if (curr === end) return true;
    let neighbors = getNeighbors(curr);
    for (let n of neighbors) {
      if (!n.isVisited) {
        n.isVisited = true;
        n.previousNode = curr;
        queue.push(n);
      }
    }
  }
  return false;
}

// Bidirectional BFS - searches from both start and end simultaneously
function bidirectionalBFS(start, end, visitedNodesInOrder) {
  // Reset all nodes
  let allNodes = getAllNodes();
  allNodes.forEach((n) => {
    n.isVisited = false;
    n.previousNode = null;
    n.backwardPreviousNode = null;
    n.visitedFromStart = false;
    n.visitedFromEnd = false;
  });

  // Two queues for forward and backward search
  let forwardQueue = [start];
  let backwardQueue = [end];

  start.visitedFromStart = true;
  start.isVisited = true;
  end.visitedFromEnd = true;
  end.isVisited = true;

  let meetingNode = null;

  while (forwardQueue.length > 0 && backwardQueue.length > 0) {
    // Expand forward search (from start) - one level at a time
    let forwardSize = forwardQueue.length;
    for (let i = 0; i < forwardSize; i++) {
      let curr = forwardQueue.shift();
      visitedNodesInOrder.push(curr);

      let neighbors = getNeighbors(curr);
      for (let neighbor of neighbors) {
        if (!neighbor.visitedFromStart) {
          neighbor.visitedFromStart = true;
          neighbor.isVisited = true;
          neighbor.previousNode = curr;
          forwardQueue.push(neighbor);

          // Check if paths meet
          if (neighbor.visitedFromEnd) {
            meetingNode = neighbor;
            break;
          }
        }
      }
      if (meetingNode) break;
    }

    if (meetingNode) break;

    // Expand backward search (from end) - one level at a time
    let backwardSize = backwardQueue.length;
    for (let i = 0; i < backwardSize; i++) {
      let curr = backwardQueue.shift();
      visitedNodesInOrder.push(curr);

      let neighbors = getNeighbors(curr);
      for (let neighbor of neighbors) {
        if (!neighbor.visitedFromEnd) {
          neighbor.visitedFromEnd = true;
          neighbor.isVisited = true;
          neighbor.backwardPreviousNode = curr;
          backwardQueue.push(neighbor);

          // Check if paths meet
          if (neighbor.visitedFromStart) {
            meetingNode = neighbor;
            break;
          }
        }
      }
      if (meetingNode) break;
    }

    if (meetingNode) break;
  }

  // If paths met, reconstruct the full path
  if (meetingNode) {
    // Reconstruct path from start to meeting point
    let pathFromStart = [];
    let curr = meetingNode;
    while (curr !== null) {
      pathFromStart.unshift(curr);
      curr = curr.previousNode;
    }

    // Reconstruct path from meeting point to end
    let pathToEnd = [];
    curr = meetingNode;
    while (curr !== null) {
      curr = curr.backwardPreviousNode;
      if (curr !== null) {
        pathToEnd.push(curr);
      }
    }

    // Combine paths and set previousNode for animation
    let fullPath = pathFromStart.concat(pathToEnd);
    for (let i = 1; i < fullPath.length; i++) {
      fullPath[i].previousNode = fullPath[i - 1];
    }

    return true;
  }

  return false;
}

function dfs(start, end, visitedNodesInOrder) {
  let stack = [start];
  while (stack.length) {
    let curr = stack.pop();
    if (!curr.isVisited) {
      curr.isVisited = true;
      visitedNodesInOrder.push(curr);
      if (curr === end) return true;
      let neighbors = getNeighbors(curr);
      for (let n of neighbors) {
        if (!n.isVisited) {
          n.previousNode = curr;
          stack.push(n);
        }
      }
    }
  }
  return false;
}

// Brute Force DFS - finds all paths and returns the shortest one
function dfsBruteForce(start, end, visitedNodesInOrder) {
  let shortestPath = null;
  let minLength = Infinity;
  let iterations = 0;
  const MAX_ITERATIONS = 40000000; // 放宽限制并防止浏览器卡死

  // 核心优化：记录到达每个节点的最短步数，用于分支限界（Branch and Bound）
  let minCostToReach = new Map();

  function findAllPaths(current, currentPath, visited) {
    if (iterations > MAX_ITERATIONS) return;
    iterations++;

    // 剪枝 1: 如果当前路径长度已经≥已知终点最短路径长度，没必要再搜索（不可能是更优解）
    if (currentPath.length >= minLength) return;

    // 剪枝 2: 如果当前节点已经被一条更短或相等步数的路径访问过，直接剪枝（不改变无启发式的穷举本质）
    let currentCost = currentPath.length;
    let prevBestCost = minCostToReach.get(current) || Infinity;
    if (currentCost >= prevBestCost) return;
    minCostToReach.set(current, currentCost);

    // 记录路径和访问状态
    currentPath.push(current);
    visited.add(current);

    // 加入动画列表，标记为已被访问
    if (!current.isVisited) {
      current.isVisited = true;
      visitedNodesInOrder.push(current);
    }

    if (current === end) {
      // 找到了一条更短的路径
      minLength = currentPath.length;
      shortestPath = [...currentPath];
    } else {
      let neighbors = getNeighbors(current);

      for (let neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          findAllPaths(neighbor, currentPath, visited);
        }
      }
    }

    // 回溯恢复状态
    visited.delete(current);
    currentPath.pop();
  }

  let visited = new Set();
  findAllPaths(start, [], visited);

  if (!shortestPath) {
    console.log(`未找到路径，或者搜索达到上限限制 (迭代次数: ${iterations})`);
    return false;
  }

  // 将找到的最短路径关联起来，以便最后能被绘制出来
  for (let i = 1; i < shortestPath.length; i++) {
    shortestPath[i].previousNode = shortestPath[i - 1];
  }

  console.log(`搜索完毕，最短路径长度: ${shortestPath.length - 1}，总计探索步数: ${iterations}`);
  return true;
}

function dijkstra(start, end, visitedNodesInOrder) {
  let unvisitedNodes = getAllNodes();
  unvisitedNodes.forEach((n) => (n.distance = Infinity));
  start.distance = 0;
  while (unvisitedNodes.length) {
    unvisitedNodes.sort((a, b) => a.distance - b.distance);
    let closestNode = unvisitedNodes.shift();
    if (closestNode.isWall) continue;
    if (closestNode.distance === Infinity) return false;
    closestNode.isVisited = true;
    visitedNodesInOrder.push(closestNode);
    if (closestNode === end) return true;
    updateUnvisitedNeighbors(closestNode);
  }
  return false;
}

function astar(start, end, visitedNodesInOrder, weight) {
  let unvisitedNodes = getAllNodes();
  unvisitedNodes.forEach((n) => {
    n.g = Infinity;
    n.f = Infinity;
  });
  start.g = 0;
  start.f = manhattan(start, end) * weight;
  start.isVisited = true;
  let openSet = [start];

  while (openSet.length) {
    openSet.sort((a, b) => a.f - b.f);
    let curr = openSet.shift();
    if (curr.isWall) continue;
    curr.isVisited = true;
    visitedNodesInOrder.push(curr);
    if (curr === end) return true;

    let neighbors = getNeighbors(curr);
    for (let neighbor of neighbors) {
      if (neighbor.isVisited) continue;
      let tempG = curr.g + 1;
      if (tempG < neighbor.g || !openSet.includes(neighbor)) {
        neighbor.g = tempG;
        neighbor.f = tempG + manhattan(neighbor, end) * weight;
        neighbor.previousNode = curr;
        if (!openSet.includes(neighbor)) openSet.push(neighbor);
      }
    }
  }
  return false;
}

// Greedy Best-First Search Algorithm
function greedyBestFirstSearch(start, end, visitedNodesInOrder) {
  let unvisitedNodes = getAllNodes();
  unvisitedNodes.forEach((n) => {
    n.h = manhattan(n, end);
  });

  start.isVisited = true;
  let openSet = [start];

  while (openSet.length) {
    // Sort by heuristic value only (greedy approach)
    openSet.sort((a, b) => a.h - b.h);
    let curr = openSet.shift();

    if (curr.isWall) continue;
    curr.isVisited = true;
    visitedNodesInOrder.push(curr);

    if (curr === end) return true;

    let neighbors = getNeighbors(curr);
    for (let neighbor of neighbors) {
      if (!neighbor.isVisited && !openSet.includes(neighbor)) {
        neighbor.previousNode = curr;
        openSet.push(neighbor);
      }
    }
  }
  return false;
}

function manhattan(a, b) {
  return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

function updateUnvisitedNeighbors(node) {
  let neighbors = getNeighbors(node);
  for (let neighbor of neighbors) {
    if (!neighbor.isVisited) {
      neighbor.distance = node.distance + 1;
      neighbor.previousNode = node;
    }
  }
}

function getAllNodes() {
  let nodes = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      nodes.push(grid[r][c]);
    }
  }
  return nodes;
}

function getShortestPath(end) {
  let path = [];
  let curr = end;
  while (curr != null) {
    path.unshift(curr);
    curr = curr.previousNode;
  }
  return path[0] === grid[startNode.row][startNode.col] ? path : [];
}

function animateAlgorithm(visitedNodes, path) {
  isAnimating = true;
  let speedValue = parseInt(document.getElementById("speed").value);
  // Map speed 1-100 to delay 50-1ms (higher speed = lower delay)
  let speed = Math.max(1, 51 - Math.floor(speedValue / 2));

  for (let i = 0; i <= visitedNodes.length; i++) {
    if (i === visitedNodes.length) {
      setTimeout(() => {
        animatePath(path);
      }, i * speed);
      return;
    }
    setTimeout(() => {
      let node = visitedNodes[i];
      if ((node.row !== startNode.row || node.col !== startNode.col) && (node.row !== endNode.row || node.col !== endNode.col)) {
        document.getElementById(`node-${node.row}-${node.col}`).classList.add("node-visited");
      }
    }, i * speed);
  }
}

function animatePath(path) {
  let speed = 50;
  for (let i = 0; i < path.length; i++) {
    setTimeout(() => {
      let node = path[i];
      if ((node.row !== startNode.row || node.col !== startNode.col) && (node.row !== endNode.row || node.col !== endNode.col)) {
        document.getElementById(`node-${node.row}-${node.col}`).classList.add("node-path");
      }
    }, i * speed);
  }
  setTimeout(() => {
    isAnimating = false;
  }, path.length * speed);
}

document.getElementById("btn-start").addEventListener("click", () => {
  if (isAnimating) return;
  runAlgorithm(document.getElementById("algorithm").value);
});
document.getElementById("btn-clear-board").addEventListener("click", () => {
  if (!isAnimating) clearBoard();
});
document.getElementById("btn-clear-path").addEventListener("click", () => {
  if (!isAnimating) clearPath();
});
document.getElementById("btn-generate-maze").addEventListener("click", () => {
  if (!isAnimating) generateRandomMaze();
});
document.getElementById("btn-apply-size").addEventListener("click", () => {
  if (!isAnimating) updateGridSize();
});

// Allow Enter key to apply grid size
document.getElementById("grid-rows").addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !isAnimating) updateGridSize();
});
document.getElementById("grid-cols").addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !isAnimating) updateGridSize();
});

// Init purely local state bindings and UI
document.body.addEventListener("mouseleave", handleMouseUp);
initializeGrid();
